// Ternary single-parent Q/K/output-gate/V projection.
//
// The parent stores rows in physical order query(6144), key(1024), gate(6144),
// value(1024) while the public output order is q, gate, k, v. Each role is a
// zero-copy row slice of the same parent, so every projection reuses the ternary
// linear kernel and no extra weight storage or workspace is needed.

#include "ops/linear/q2/q2_dispatch.h"

#include "core/tensor.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Slice `rows` consecutive rows starting at `first_row` out of a row-split parent.
Weight slice_rows(const Weight& parent, std::int32_t first_row, std::int32_t rows) {
    if (parent.layout != QuantLayout::RowSplit || parent.k <= 0 ||
        parent.k % parent.group_size != 0) {
        throw std::invalid_argument("q2 attn_input_proj: parent must be row-split");
    }
    const std::uint64_t groups = static_cast<std::uint64_t>(parent.k / parent.group_size);
    Weight out                = parent;
    out.n                     = rows;
    out.shape[0] = out.padded_shape[0] = rows;
    out.qdata = static_cast<const std::uint8_t*>(parent.qdata) +
                static_cast<std::uint64_t>(first_row) * groups * 32u;
    out.scales = static_cast<const std::uint8_t*>(parent.scales) +
                 static_cast<std::uint64_t>(first_row) * groups * 2u;
    return out;
}

} // namespace

void q2_attn_input_proj(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                        Tensor& v, WorkspaceArena* workspace, cudaStream_t stream) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    constexpr std::int32_t kRows   = 14336;
    if (weight.qtype != QType::Q2_G128_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 128 || weight.group != 128 ||
        weight.n != kRows || weight.k != kHidden || weight.shape[0] != kRows ||
        weight.shape[1] != kHidden || weight.padded_shape[0] != kRows ||
        weight.padded_shape[1] != kHidden || weight.qdata == nullptr ||
        weight.scales == nullptr || weight.qhigh != nullptr) {
        throw std::invalid_argument("q2 attn_input_proj: invalid parent weight");
    }
    const std::int32_t cols = x.ne[1];
    if (cols <= 0 || q.ne[0] != kQRows || gate.ne[0] != kQRows || k.ne[0] != kKvRows ||
        v.ne[0] != kKvRows) {
        throw std::invalid_argument("q2 attn_input_proj: operand geometry mismatch");
    }

    const Weight query  = slice_rows(weight, 0, kQRows);
    const Weight key    = slice_rows(weight, kQRows, kKvRows);
    const Weight gating = slice_rows(weight, kQRows + kKvRows, kQRows);
    const Weight value  = slice_rows(weight, kQRows + kKvRows + kQRows, kKvRows);
    if (weight.hadamard_signs != nullptr && workspace == nullptr) {
        throw std::invalid_argument("q2 attn_input_proj: Hadamard workspace is required");
    }
    const Tensor transformed =
        weight.hadamard_signs ? q2_prepare_activation(x, weight, *workspace, stream) : x;

    q2_linear_launch(transformed, query, q, stream, false);
    q2_linear_launch(transformed, key, k, stream, false);
    q2_linear_launch(transformed, gating, gate, stream, false);
    q2_linear_launch(transformed, value, v, stream, false);
}

} // namespace ninfer::ops::detail
