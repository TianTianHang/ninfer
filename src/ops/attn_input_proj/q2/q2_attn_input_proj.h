#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q2_attn_input_proj(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                        Tensor& v, WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
