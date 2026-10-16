// Generic ternary (Q2_G128_FP16) A16 linear kernel.
//
// One kernel covers every (N,K) the model uses, so no per-shape instantiation is
// needed. It is a register-blocked SIMT contraction rather than an MMA route: the
// goal is a correct, complete operator family first, with the decode rules of
// `q2_g128_fp16` written down in exactly one place.
//
// Storage (row_split_k128_v1, group 128, four codes per byte, low slot first):
//   element k of a row has code c = (byte[(k%128)/4] >> 2*(k%4)) & 3, taken from
//   the row's code plane at group g = k/128; group g carries the fp16 scale at
//   scales[row * groups_per_row + g]. The code is an unsigned grid index and
//   represents (c - 1) * scale.

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
constexpr int kCodeBytesPerGroup = 32;
constexpr int kBlockRows = 64;
constexpr int kBlockCols = 64;
constexpr int kThreads = 256;
constexpr int kThreadRows = 4;
constexpr int kThreadCols = 4;
constexpr int kKTile = 128;
constexpr int kMaxGroupsPerTile = 1;  // every tile is exactly one quant group
constexpr int kStageStride = kBlockCols + 4;

__device__ __forceinline__ float grid_value(std::uint8_t byte, int slot) {
    return static_cast<float>(((byte >> (2 * slot)) & 0x03) - 1);
}

__global__ void q2_a16_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                   const std::uint8_t* __restrict__ base,
                                   const __half* __restrict__ scales,
                                   __nv_bfloat16* __restrict__ out, int n, int k, int padded_k,
                                   int out_row_stride, bool accumulate) {
    constexpr int kWarpsPerBlock = 16;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= n) { return; }
    const int groups_per_row = padded_k / kGroup;
    float sum = 0.0f;
    for (int kk = lane; kk < k; kk += 32) {
        const int group = kk / kGroup;
        const int local = kk % kGroup;
        const auto byte = base[static_cast<std::size_t>(row) * groups_per_row * kCodeBytesPerGroup +
                               static_cast<std::size_t>(group) * kCodeBytesPerGroup + local / 4];
        const float weight = grid_value(byte, local % 4) *
                             __half2float(scales[static_cast<std::size_t>(row) * groups_per_row + group]);
        sum = fmaf(weight, __bfloat162float(x[kk]), sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        const auto at = static_cast<std::size_t>(row) * out_row_stride;
        const float value = accumulate ? sum + __bfloat162float(out[at]) : sum;
        out[at] = __float2bfloat16(value);
    }
}

__global__ void q2_a16_simt_kernel(const __nv_bfloat16* __restrict__ x,
                                   const std::uint8_t* __restrict__ base,
                                   const __half* __restrict__ scales,
                                   __nv_bfloat16* __restrict__ out, int n, int k, int padded_k, int t,
                                   int out_row_stride, int out_token_stride, bool accumulate) {
    const int row0 = blockIdx.y * kBlockRows;
    const int col0 = blockIdx.x * kBlockCols;
    const int tid = threadIdx.x;
    const int groups_per_row = padded_k / kGroup;

    const int local_row = tid / (kBlockCols / kThreadCols) * kThreadRows;
    const int local_col = tid % (kBlockCols / kThreadCols) * kThreadCols;

    float acc[kThreadRows][kThreadCols];
    for (int i = 0; i < kThreadRows; ++i) {
        for (int j = 0; j < kThreadCols; ++j) { acc[i][j] = 0.0f; }
    }

    __shared__ __nv_bfloat16 stage[kKTile][kStageStride];
    __shared__ std::uint8_t codes[kBlockRows][kMaxGroupsPerTile * kCodeBytesPerGroup];
    __shared__ float group_scales[kMaxGroupsPerTile][kBlockRows];

    for (int k0 = 0; k0 < k; k0 += kKTile) {
        const int first_group = k0 / kGroup;
        const int last_group = (min(k0 + kKTile, k) - 1) / kGroup;
        const int groups_here = last_group - first_group + 1;

        // Stage the activation tile x[k0, k0+kKTile) x [col0, col0+kBlockCols).
        for (int index = tid; index < kKTile * kBlockCols; index += kThreads) {
            const int kk = index / kBlockCols;
            const int cc = index % kBlockCols;
            const int global_k = k0 + kk;
            const int global_c = col0 + cc;
            float value = 0.0f;
            if (global_k < k && global_c < t) {
                value = __bfloat162float(x[static_cast<std::size_t>(global_c) * k + global_k]);
            }
            stage[kk][cc] = __float2bfloat16(value);
        }

        // Stage the weight rows this tile touches, row-major: the k bytes a thread
        // consumes are then contiguous, and the scale is fetched once per group.
        for (int index = tid; index < kBlockRows * groups_here; index += kThreads) {
            const int local = index / groups_here;
            const int g = index % groups_here;
            const int row = row0 + local;
            if (row >= n) { continue; }
            const auto* src = base +
                static_cast<std::size_t>(row) * groups_per_row * kCodeBytesPerGroup +
                static_cast<std::size_t>(first_group + g) * kCodeBytesPerGroup;
            std::uint8_t* dst = &codes[local][g * kCodeBytesPerGroup];
            for (int i = 0; i < kCodeBytesPerGroup; ++i) { dst[i] = src[i]; }
        }
        for (int index = tid; index < groups_here * kBlockRows; index += kThreads) {
            const int g = index / kBlockRows;
            const int local = index % kBlockRows;
            const int row = row0 + local;
            group_scales[g][local] =
                row < n ? __half2float(
                              scales[static_cast<std::size_t>(row) * groups_per_row +
                                     first_group + g])
                        : 0.0f;
        }
        __syncthreads();

        for (int kk = 0; kk < kKTile; ++kk) {
            const int global_k = k0 + kk;
            if (global_k >= k) { break; }
            const int g = global_k / kGroup - first_group;
            const int lane_in_group = global_k % kGroup;
            const int byte_in_group = lane_in_group / 4;
            const int slot = lane_in_group % 4;

            float a[kThreadCols];
            for (int j = 0; j < kThreadCols; ++j) {
                a[j] = __bfloat162float(stage[kk][local_col + j]);
            }

            float b[kThreadRows];
            for (int i = 0; i < kThreadRows; ++i) {
                const int local = local_row + i;
                const int row = row0 + local;
                if (row < n) {
                    b[i] = grid_value(codes[local][g * kCodeBytesPerGroup + byte_in_group],
                                      slot) *
                           group_scales[g][local];
                } else {
                    b[i] = 0.0f;
                }
            }

            for (int i = 0; i < kThreadRows; ++i) {
                for (int j = 0; j < kThreadCols; ++j) { acc[i][j] = fmaf(b[i], a[j], acc[i][j]); }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < kThreadRows; ++i) {
        const int row = row0 + local_row + i;
        if (row >= n) { continue; }
        for (int j = 0; j < kThreadCols; ++j) {
            const int col = col0 + local_col + j;
            if (col >= t) { continue; }
            const std::size_t at = static_cast<std::size_t>(col) * out_token_stride +
                                   static_cast<std::size_t>(row) * out_row_stride;
            const float value =
                accumulate ? acc[i][j] + __bfloat162float(out[at]) : acc[i][j];
            out[at] = __float2bfloat16(value);
        }
    }
}

} // namespace

void q2_linear_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
                      bool accumulate) {
    const int n = out.ne[0];
    const int t = out.ne[1];
    const int k = x.ne[0];
    const int padded_k = weight.padded_shape[1];
    const auto* base = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* scales = static_cast<const __half*>(weight.scales);
    const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data);
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data);

    const int out_row_stride = static_cast<int>(out.nb[0] / sizeof(__nv_bfloat16));
    const int out_token_stride = static_cast<int>(out.nb[1] / sizeof(__nv_bfloat16));

    if (t == 1) {
        constexpr int kWarpsPerBlock = 16;
        q2_a16_gemv_kernel<<<(n + kWarpsPerBlock - 1) / kWarpsPerBlock,
                              kWarpsPerBlock * 32, 0, stream>>>(xp, base, scales, op, n, k,
                                                               padded_k, out_row_stride,
                                                               accumulate);
        CUDA_CHECK(cudaGetLastError());
    } else {
        const dim3 grid((t + kBlockCols - 1) / kBlockCols,
                        (n + kBlockRows - 1) / kBlockRows);
        q2_a16_simt_kernel<<<grid, kThreads, 0, stream>>>(xp, base, scales, op, n, k, padded_k, t,
                                                          out_row_stride, out_token_stride,
                                                          accumulate);
        CUDA_CHECK(cudaGetLastError());
    }
}

void q2_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 WorkspaceArena* workspace, cudaStream_t stream) {
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument("q2 linear: unsupported policy");
    }
    if (weight.qtype != QType::Q2_G128_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.ndim != 2 || weight.group_size != kGroup ||
        weight.group != kGroup || weight.shape[0] != weight.n || weight.shape[1] != weight.k ||
        weight.padded_shape[0] != weight.n || weight.padded_shape[1] != weight.k ||
        weight.qdata == nullptr || weight.qhigh != nullptr || weight.scales == nullptr) {
        throw std::invalid_argument("q2 linear: weight must be q2_g128_fp16 row-split");
    }
    if (out.ne[0] != weight.n || x.ne[0] != weight.k) {
        throw std::invalid_argument("q2 linear: weight geometry differs from its operands");
    }
    if (weight.hadamard_signs != nullptr) {
        if (workspace == nullptr) {
            throw std::invalid_argument("q2 linear: Hadamard transform requires workspace");
        }
        auto scope = workspace->scope();
        Tensor transformed = q2_prepare_activation(x, weight, *workspace, stream);
        q2_linear_launch(transformed, weight, out, stream, false);
    } else {
        q2_linear_launch(x, weight, out, stream, false);
    }
}

} // namespace ninfer::ops::detail
