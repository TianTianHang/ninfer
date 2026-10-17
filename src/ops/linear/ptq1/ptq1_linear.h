#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime_api.h>

namespace ninfer::ops::detail {

void ptq1_linear_launch(const Tensor& x, const Weight& weight, Tensor& out,
                        cudaStream_t stream, bool accumulate = false);

} // namespace ninfer::ops::detail
