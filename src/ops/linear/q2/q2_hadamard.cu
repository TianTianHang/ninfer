#include "ops/linear/q2/q2_dispatch.h"

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "core/arena.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHadamardBlock = 1024;
constexpr int kHadamardThreads = 256;
constexpr float kHadamardScale = 1.0f / 32.0f;

__global__ void q2_hadamard_kernel(const __nv_bfloat16* __restrict__ input,
                                   __nv_bfloat16* __restrict__ output,
                                   const std::int32_t* __restrict__ signs,
                                   int width, int tokens, int input_row_stride,
                                   int input_token_stride, int perm_hd, int perm_nk,
                                   int perm_rep, bool inverse) {
    const int token = blockIdx.x;
    const int block = blockIdx.y;
    const int lane = threadIdx.x;
    if (token >= tokens) { return; }

    __shared__ float first[kHadamardBlock];
    __shared__ float second[kHadamardBlock];
    float* src = first;
    float* dst = second;

    const int base = block * kHadamardBlock;
    for (int i = lane; i < kHadamardBlock; i += kHadamardThreads) {
        const int output_feature = base + i;
        int input_feature = output_feature;
        if (perm_rep > 1) {
            const int hd = output_feature % perm_hd;
            const int temp = output_feature / perm_hd;
            const int rep = temp % perm_rep;
            const int nk = temp / perm_rep;
            input_feature = hd + perm_hd * (nk + perm_nk * rep);
        }
        const float value = __bfloat162float(
            input[static_cast<std::int64_t>(token) * input_token_stride +
                  static_cast<std::int64_t>(input_feature) * input_row_stride]);
        // Forward folded weights consume H(sign*x). Inverse-folded token embeddings
        // consume sign(H(z)); the two operations are deliberately not reordered.
        src[i] = inverse ? value : value * static_cast<float>(signs[output_feature]);
    }
    __syncthreads();

    for (int stride = 1; stride < kHadamardBlock; stride <<= 1) {
        for (int i = lane; i < kHadamardBlock; i += kHadamardThreads) {
            const int partner = i ^ stride;
            const float a = src[i];
            const float b = src[partner];
            dst[i] = (i & stride) ? (b - a) : (a + b);
        }
        __syncthreads();
        float* swap = src;
        src = dst;
        dst = swap;
    }

    for (int i = lane; i < kHadamardBlock; i += kHadamardThreads) {
        const int feature = base + i;
        float value = src[i] * kHadamardScale;
        if (inverse) { value *= static_cast<float>(signs[feature]); }
        output[static_cast<std::int64_t>(token) * width + feature] = __float2bfloat16_rn(value);
    }
}

} // namespace

void q2_hadamard_transform(const Tensor& input, Tensor& output, const std::int32_t* signs,
                           int block_size, int perm_hd, int perm_nk, int perm_rep,
                           bool inverse, cudaStream_t stream) {
    if (block_size != kHadamardBlock || !signs || input.ne[0] != output.ne[0] ||
        input.ne[1] != output.ne[1] || input.ne[0] % block_size != 0 ||
        (perm_rep > 1 &&
         (perm_hd <= 0 || perm_nk <= 0 || perm_hd * perm_nk * perm_rep != input.ne[0]))) {
        throw std::invalid_argument("q2 Hadamard transform: incompatible geometry");
    }
    const int width = input.ne[0];
    const int tokens = input.ne[1];
    const int blocks = width / block_size;
    const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(blocks));
    q2_hadamard_kernel<<<grid, kHadamardThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data), static_cast<__nv_bfloat16*>(output.data),
        signs, width, tokens, static_cast<int>(input.nb[0] / sizeof(__nv_bfloat16)),
        static_cast<int>(input.nb[1] / sizeof(__nv_bfloat16)), perm_hd, perm_nk, perm_rep,
        inverse);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t q2_hadamard_workspace_bytes(int width, int tokens) {
    if (width <= 0 || tokens <= 0 || width % kHadamardBlock != 0) {
        throw std::invalid_argument("q2 Hadamard workspace: invalid geometry");
    }
    return static_cast<std::size_t>(width) * tokens * sizeof(__nv_bfloat16);
}

Tensor q2_prepare_activation(const Tensor& input, const Weight& weight,
                             WorkspaceArena& workspace, cudaStream_t stream) {
    if ((weight.qtype != QType::Q2_G128_FP16 && weight.qtype != QType::PTQ1_G128_FP16) ||
        weight.hadamard_signs == nullptr || weight.hadamard_block_size == 0) {
        return input;
    }
    Tensor output = workspace.alloc(DType::BF16, {input.ne[0], input.ne[1]});
    q2_hadamard_transform(input, output, weight.hadamard_signs, weight.hadamard_block_size,
                           weight.hadamard_perm_hd, weight.hadamard_perm_nk,
                           weight.hadamard_perm_rep, false, stream);
    return output;
}

} // namespace ninfer::ops::detail
