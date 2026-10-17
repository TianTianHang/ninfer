// Ternary (Q2_G128_FP16) embedding-table gather.
//
// Row lookup on a row_split_k128_v1 table: element k of a row has code
// c = (byte[(k%128)/4] >> 2*(k%4)) & 3 in group g = k/128; group g carries
// the fp16 scale at scales[row * groups_per_row + g].

#include "ops/launcher/embed_gather.h"

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
constexpr int kThreads = 512;

struct block_ptq1_0 {
    std::uint8_t qs[24];
    std::uint8_t qh[2];
    __half d;
};
static_assert(sizeof(block_ptq1_0) == 28, "invalid PTQ1 block size");

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

__global__ void embed_gather_q2_kernel(const std::int32_t* __restrict__ ids,
                                       const std::uint8_t* __restrict__ codes,
                                       const __half* __restrict__ scales,
                                       __nv_bfloat16* __restrict__ out,
                                       const std::int32_t* __restrict__ signs,
                                       int rows, int hidden, int tokens, bool ptq1) {
    constexpr int kHadamard = 1024;
    constexpr float kScale = 1.0f / 32.0f;
    const int token = blockIdx.x;
    if (token >= tokens) { return; }
    const int row = ids[token];
    if (row < 0 || row >= rows) { return; }
    const int groups_per_row = hidden / kGroup;
    const std::uint8_t* row_codes =
        codes + static_cast<std::size_t>(row) * groups_per_row * (ptq1 ? 28 : kCodeBytesPerGroup);
    const __half* row_scales = scales == nullptr ? nullptr :
        scales + static_cast<std::size_t>(row) * groups_per_row;
    const auto* row_ptq1 = reinterpret_cast<const block_ptq1_0*>(row_codes);
    __nv_bfloat16* dst = out + static_cast<std::size_t>(token) * hidden;
    __shared__ float first[kHadamard];
    __shared__ float second[kHadamard];
    for (int block = 0; block < hidden / kHadamard; ++block) {
        for (int i = threadIdx.x; i < kHadamard; i += kThreads) {
            const int feature = block * kHadamard + i;
            const int group = feature / kGroup;
            const int lane = feature % kGroup;
            if (ptq1) {
                const auto& ptq_block = row_ptq1[group];
                first[i] = static_cast<float>(ptq1_0_trit(&ptq_block, lane)) *
                           __half2float(ptq_block.d);
            } else {
                const std::uint8_t byte = row_codes[group * kCodeBytesPerGroup + lane / 4];
                const int code = (byte >> (2 * (lane % 4))) & 0x03;
                first[i] = static_cast<float>(code - 1) * __half2float(row_scales[group]);
            }
        }
        __syncthreads();
        float* src = first;
        float* dst_stage = second;
        for (int stride = 1; stride < kHadamard; stride <<= 1) {
            for (int i = threadIdx.x; i < kHadamard; i += kThreads) {
                const float a = src[i];
                const float b = src[i ^ stride];
                dst_stage[i] = (i & stride) ? b - a : a + b;
            }
            __syncthreads();
            float* tmp = src;
            src = dst_stage;
            dst_stage = tmp;
        }
        for (int i = threadIdx.x; i < kHadamard; i += kThreads) {
            const int feature = block * kHadamard + i;
            float value = src[i] * kScale;
            // token_embd.weight is inverse-folded: Prism restores H(z), then
            // applies the explicit sign vector in the output feature order.
            if (signs) { value *= static_cast<float>(signs[feature]); }
            dst[feature] = __float2bfloat16_rn(value);
        }
        __syncthreads();
    }
}

} // namespace

void embed_gather_q2_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream) {
    if (table.qtype != QType::Q2_G128_FP16 || table.layout != QuantLayout::RowSplit ||
        table.group_size != 128 || table.k % kGroup != 0 || table.qdata == nullptr ||
        table.scales == nullptr) {
        throw std::invalid_argument("embed_q2: table must be q2_g128_fp16 row-split");
    }
    const int tokens = static_cast<int>(out.ne[1]);
    if (tokens <= 0) { return; }
    const int rows = table.n;
    const int hidden = table.k;
    embed_gather_q2_kernel<<<tokens, kThreads, 0, stream>>>(
        reinterpret_cast<const std::int32_t*>(ids.data),
        static_cast<const std::uint8_t*>(table.qdata), static_cast<const __half*>(table.scales),
        reinterpret_cast<__nv_bfloat16*>(out.data),
        table.hadamard_signs, rows, hidden, tokens, false);
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_ptq1_launch(const Tensor& ids, const Weight& table, Tensor& out,
                              cudaStream_t stream) {
    if (table.qtype != QType::PTQ1_G128_FP16 || table.layout != QuantLayout::RowSplit ||
        table.group_size != 128 || table.k % kGroup != 0 || table.qdata == nullptr ||
        table.scales != nullptr || table.qhigh != nullptr) {
        throw std::invalid_argument("embed_ptq1: table must be ptq1_g128_fp16 row-split");
    }
    const int tokens = static_cast<int>(out.ne[1]);
    if (tokens <= 0) { return; }
    const int rows = table.n;
    const int hidden = table.k;
    embed_gather_q2_kernel<<<tokens, kThreads, 0, stream>>>(
        reinterpret_cast<const std::int32_t*>(ids.data),
        static_cast<const std::uint8_t*>(table.qdata), nullptr,
        reinterpret_cast<__nv_bfloat16*>(out.data), table.hadamard_signs,
        rows, hidden, tokens, true);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
