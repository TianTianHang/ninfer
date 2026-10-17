// Native PTQ1_0 A16 linear contraction.
//
// The block decoder follows llama.cpp's block_ptq1_0 and ptq1_0_trit
// implementation exactly. PTQ1 keeps the scale in each 28-byte block, so the
// kernel consumes the source payload directly instead of expanding it to Q2.

#include "ops/linear/ptq1/ptq1_linear.h"

#include "core/device.h"

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

__global__ void ptq1_a16_warp_kernel(const __nv_bfloat16* __restrict__ x,
                                     const std::uint8_t* __restrict__ weights,
                                     __nv_bfloat16* __restrict__ out, int n, int k, int t,
                                     int x_token_stride, int out_token_stride,
                                     bool accumulate) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const std::size_t work = static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const std::size_t total = static_cast<std::size_t>(n) * t;
    if (work >= total) { return; }
    const int row = static_cast<int>(work % n);
    const int token = static_cast<int>(work / n);
    const int blocks_per_row = k / kGroup;
    const auto* row_weights = reinterpret_cast<const block_ptq1_0*>(weights) +
                               static_cast<std::size_t>(row) * blocks_per_row;
    const auto* activation = x + static_cast<std::size_t>(token) * x_token_stride;
    float sum = 0.0f;
    for (int group = 0; group < blocks_per_row; ++group) {
        const auto* block = row_weights + group;
        const float scale = lane == 0 ? __half2float(block->d) : 0.0f;
        const float shared_scale = __shfl_sync(0xffffffffu, scale, 0);
#pragma unroll
        for (int offset = 0; offset < kGroup; offset += 32) {
            const int element = offset + lane;
            const float weight = static_cast<float>(ptq1_0_trit(block, element)) * shared_scale;
            sum = fmaf(weight, __bfloat162float(activation[group * kGroup + element]), sum);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        const std::size_t output = static_cast<std::size_t>(token) * out_token_stride + row;
        const float value = accumulate ? sum + __bfloat162float(out[output]) : sum;
        out[output] = __float2bfloat16(value);
    }
}

} // namespace

void ptq1_linear_launch(const Tensor& x, const Weight& weight, Tensor& out,
                        cudaStream_t stream, bool accumulate) {
    if (weight.qtype != QType::PTQ1_G128_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.group_size != kGroup || weight.group != kGroup || weight.qdata == nullptr ||
        weight.scales != nullptr || weight.qhigh != nullptr || weight.k <= 0 ||
        weight.k % kGroup != 0 || weight.n <= 0 || x.ne[0] != weight.k || out.ne[0] != weight.n ||
        x.ne[1] != out.ne[1] || x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("ptq1 linear: invalid weight or operand geometry");
    }
    const int t = x.ne[1];
    if (t <= 0) { return; }
    const int blocks = static_cast<int>((static_cast<std::size_t>(weight.n) * t +
                                         kWarpsPerBlock - 1) /
                                        kWarpsPerBlock);
    ptq1_a16_warp_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<__nv_bfloat16*>(out.data), weight.n, weight.k, t,
        static_cast<int>(x.nb[1] / sizeof(__nv_bfloat16)),
        static_cast<int>(out.nb[1] / sizeof(__nv_bfloat16)), accumulate);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
