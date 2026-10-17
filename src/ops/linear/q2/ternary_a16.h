#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime_api.h>

namespace ninfer::ops::detail {

// Native ternary (Q2_G128_FP16 or PTQ1_G128_FP16) A16 contraction. Validates the
// stored row-split representation and operands, then selects the private route
// for the requested token extent. `accumulate` adds into `out` instead of
// overwriting it. The activation transform for Hadamard-folded weights belongs
// to the caller.
void ternary_a16_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
                        bool accumulate);

} // namespace ninfer::ops::detail
