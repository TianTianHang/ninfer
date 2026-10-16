#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t q2_linear_swiglu_workspace_bytes(std::int32_t gate_up_rows,
                                                           std::int32_t input_rows,
                                                           std::int32_t max_tokens);

void q2_linear_swiglu(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& ws,
                      cudaStream_t stream);

} // namespace ninfer::ops::detail
