// Ternary SwiGLU projection.
//
// The weight is one fused parent with gate rows [0,M) then up rows [M,2M).
// PTQ1 uses a fused contraction so the gate/up projection does not materialize
// an intermediate BF16 tensor before the SiLU product.

#include "ops/linear/q2/q2_dispatch.h"

#include "core/device.h"
#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kGroup = 128;
constexpr int kBlockBytes = 28;
constexpr int kThreads = 256;
constexpr int kWarpsPerBlock = kThreads / 32;

struct block_ptq1_0 {
    std::uint8_t qs[24];
    std::uint8_t qh[2];
    __half d;
};
static_assert(sizeof(block_ptq1_0) == kBlockBytes, "invalid PTQ1 block size");

__device__ __forceinline__ int ptq1_0_trit(const block_ptq1_0* x, int e) {
    std::uint8_t b;
    int n;
    if (e < 80) {
        b = x->qs[e & 15];
        n = e >> 4;
    } else if (e < 120) {
        const int t = e - 80;
        b = x->qs[16 + (t & 7)];
        n = t >> 3;
    } else {
        const int t = e - 120;
        b = x->qh[t & 1];
        n = t >> 1;
    }
    std::uint32_t value = b;
    for (int i = 0; i < 4; ++i) {
        if (i < n) { value = (value * 3u) & 0xffu; }
    }
    return static_cast<int>((value * 3u) >> 8) - 1;
}

__global__ void ptq1_swiglu_kernel(const __nv_bfloat16* __restrict__ x,
                                    const std::uint8_t* __restrict__ weights,
                                    __nv_bfloat16* __restrict__ out, int rows, int k, int t,
                                    int x_token_stride, int out_token_stride) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const std::size_t work = static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const std::size_t total = static_cast<std::size_t>(rows) * t;
    if (work >= total) { return; }
    const int row = static_cast<int>(work % rows);
    const int token = static_cast<int>(work / rows);
    const int groups_per_row = k / kGroup;
    const auto* parent = reinterpret_cast<const block_ptq1_0*>(weights);
    const auto* gate_row = parent + static_cast<std::size_t>(row) * groups_per_row;
    const auto* up_row = parent + static_cast<std::size_t>(rows + row) * groups_per_row;
    const auto* activation = x + static_cast<std::size_t>(token) * x_token_stride;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int group = 0; group < groups_per_row; ++group) {
        const auto* gate_block = gate_row + group;
        const auto* up_block = up_row + group;
        const float gate_scale = lane == 0 ? __half2float(gate_block->d) : 0.0f;
        const float up_scale = lane == 0 ? __half2float(up_block->d) : 0.0f;
        const float shared_gate_scale = __shfl_sync(0xffffffffu, gate_scale, 0);
        const float shared_up_scale = __shfl_sync(0xffffffffu, up_scale, 0);
#pragma unroll
        for (int offset = 0; offset < kGroup; offset += 32) {
            const int element = offset + lane;
            const float a = __bfloat162float(activation[group * kGroup + element]);
            gate_sum = fmaf(static_cast<float>(ptq1_0_trit(gate_block, element)) *
                                shared_gate_scale,
                            a, gate_sum);
            up_sum = fmaf(static_cast<float>(ptq1_0_trit(up_block, element)) * shared_up_scale,
                          a, up_sum);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down_sync(0xffffffffu, gate_sum, offset);
        up_sum += __shfl_down_sync(0xffffffffu, up_sum, offset);
    }
    if (lane == 0) {
        const float gate = __bfloat162float(__float2bfloat16(gate_sum));
        const float up = __bfloat162float(__float2bfloat16(up_sum));
        const float silu = gate / (1.0f + expf(-gate));
        out[static_cast<std::size_t>(token) * out_token_stride + row] =
            __float2bfloat16(silu * up);
    }
}

__global__ void swiglu_reduce_kernel(const __nv_bfloat16* __restrict__ gate_up,
                                     __nv_bfloat16* __restrict__ out, int rows, int t) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * t;
    if (index >= total) { return; }
    const int row = static_cast<int>(index % rows);
    const int col = static_cast<int>(index / rows);
    const float gate = __bfloat162float(gate_up[static_cast<std::size_t>(col) * (2 * rows) + row]);
    const float up =
        __bfloat162float(gate_up[static_cast<std::size_t>(col) * (2 * rows) + rows + row]);
    const float silu = gate / (1.0f + expf(-gate));
    out[static_cast<std::size_t>(col) * rows + row] = __float2bfloat16(silu * up);
}

} // namespace

std::size_t q2_linear_swiglu_workspace_bytes(std::int32_t gate_up_rows,
                                             std::int32_t input_rows,
                                             std::int32_t max_tokens) {
    if (gate_up_rows <= 0 || input_rows <= 0 || max_tokens <= 0) {
        throw std::invalid_argument("q2 linear_swiglu: rows and tokens must be positive");
    }
    return static_cast<std::size_t>(gate_up_rows + input_rows) *
           static_cast<std::size_t>(max_tokens) * 2u;
}

void q2_linear_swiglu(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& ws,
                      cudaStream_t stream) {
    if ((weight.qtype != QType::Q2_G128_FP16 && weight.qtype != QType::PTQ1_G128_FP16) ||
        weight.layout != QuantLayout::RowSplit || weight.group_size != 128 ||
        weight.group != 128 || weight.n % 2 != 0 || weight.qdata == nullptr ||
        weight.qhigh != nullptr || (weight.qtype == QType::Q2_G128_FP16 && weight.scales == nullptr) ||
        (weight.qtype == QType::PTQ1_G128_FP16 && weight.scales != nullptr)) {
        throw std::invalid_argument("ternary linear_swiglu: weight must be q2 or ptq1 row-split");
    }
    const int rows = weight.n / 2;
    const int t = x.ne[1];
    if (rows <= 0 || t <= 0 || out.ne[0] != rows || out.ne[1] != t) {
        throw std::invalid_argument("q2 linear_swiglu: operand geometry disagrees with the weight");
    }

    auto scope = ws.scope();
    Tensor transformed = q2_prepare_activation(x, weight, ws, stream);
    if (weight.qtype == QType::PTQ1_G128_FP16) {
        if (weight.k <= 0 || weight.k % kGroup != 0 || x.ne[0] != weight.k ||
            transformed.ne[0] != weight.k || transformed.dtype != DType::BF16 ||
            out.dtype != DType::BF16) {
            throw std::invalid_argument("ptq1 linear_swiglu: invalid operand geometry");
        }
        const int blocks = static_cast<int>((static_cast<std::size_t>(rows) * t +
                                             kWarpsPerBlock - 1) /
                                            kWarpsPerBlock);
        ptq1_swiglu_kernel<<<blocks, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(transformed.data),
            static_cast<const std::uint8_t*>(weight.qdata), static_cast<__nv_bfloat16*>(out.data),
            rows, weight.k, t, static_cast<int>(transformed.nb[1] / sizeof(__nv_bfloat16)),
            static_cast<int>(out.nb[1] / sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    Tensor projected = ws.alloc(DType::BF16, {weight.n, t});
    auto* gate_up = reinterpret_cast<const __nv_bfloat16*>(projected.data);
    q2_linear_launch(transformed, weight, projected, stream, false);

    const std::size_t total = static_cast<std::size_t>(rows) * t;
    const int threads = 256;
    const std::size_t blocks = (total + threads - 1) / threads;
    swiglu_reduce_kernel<<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
        gate_up, reinterpret_cast<__nv_bfloat16*>(out.data), rows, t);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
