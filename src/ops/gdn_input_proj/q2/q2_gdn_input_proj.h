#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q2_gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                       Tensor& qkv, Tensor& z, WorkspaceArena& workspace,
                       cudaStream_t stream);

void q2_gdn_input_proj_single(const Tensor& x, const Weight& parent, Tensor& qkv, Tensor& z,
                              WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
