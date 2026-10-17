// Ternary linear dispatch (Q2_G128_FP16 / PTQ1_G128_FP16).
//
// The contraction itself is the shared ternary A16 implementation; this file owns
// the semantic dispatch: Hadamard activation preparation for folded weights, the
// call-scoped workspace, and the direct launch entry that fused projections use.

#include "ops/linear/q2/q2_dispatch.h"

#include "ops/linear/q2/ternary_a16.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <stdexcept>

namespace ninfer::ops::detail {

void q2_linear_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
                      bool accumulate) {
    ternary_a16_launch(x, weight, out, stream, accumulate);
}

void q2_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 WorkspaceArena* workspace, cudaStream_t stream) {
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument("ternary linear: unsupported policy");
    }
    if (weight.qtype != QType::Q2_G128_FP16 && weight.qtype != QType::PTQ1_G128_FP16) {
        throw std::invalid_argument("ternary linear: weight must be q2 or ptq1 row-split");
    }
    if (weight.ndim != 2 || weight.k != weight.shape[1] || weight.n != weight.shape[0] ||
        x.ne[0] != weight.k || out.ne[0] != weight.n || x.ne[1] != out.ne[1]) {
        throw std::invalid_argument("ternary linear: weight geometry differs from its operands");
    }
    if (weight.hadamard_signs != nullptr) {
        if (workspace == nullptr) {
            throw std::invalid_argument("ternary linear: Hadamard transform requires workspace");
        }
        auto scope         = workspace->scope();
        Tensor transformed = q2_prepare_activation(x, weight, *workspace, stream);
        ternary_a16_launch(transformed, weight, out, stream, false);
    } else {
        ternary_a16_launch(x, weight, out, stream, false);
    }
}

} // namespace ninfer::ops::detail
