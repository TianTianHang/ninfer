#pragma once

#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/ptq1/ptq1_linear.h"

#include <cstdint>

namespace ninfer::ops::detail {

// Ternary A16 linear. One generic kernel serves every (N,K), so selection only
// has to validate that the weight is the registered q2_g128_fp16 row-split form.
// `accumulate` adds into `out` instead of overwriting it, which is what the
// residual projection (`linear_add`) needs and saves it a second pass.
void q2_linear_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
                      bool accumulate = false);

void q2_hadamard_transform(const Tensor& input, Tensor& output, const std::int32_t* signs,
                           int block_size, int perm_hd, int perm_nk, int perm_rep,
                           bool inverse, cudaStream_t stream);
[[nodiscard]] std::size_t q2_hadamard_workspace_bytes(int width, int tokens);
[[nodiscard]] Tensor q2_prepare_activation(const Tensor& input, const Weight& weight,
                                           WorkspaceArena& workspace, cudaStream_t stream);

void q2_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
