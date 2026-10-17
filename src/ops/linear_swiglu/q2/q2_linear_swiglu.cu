// Ternary SwiGLU projection.
//
// The weight is one fused parent with gate rows [0,M) then up rows [M,2M), which is
// exactly what the generic ternary linear computes. The projection is materialized
// into caller-owned workspace and reduced by a cheap elementwise pass; fusing the
// two into one kernel that never leaves registers is the next performance step.

#include "ops/linear/q2/q2_dispatch.h"

#include "core/device.h"
#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

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
