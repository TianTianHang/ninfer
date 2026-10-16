// Ternary paired GDN input projection.
//
// qk_weight is a [4096,5120] ternary parent holding query then key rows, and
// value_z_weight is a [12288,5120] ternary parent holding value then z rows. Both
// are contiguous, so each role is a zero-copy row slice and the work reduces to
// four ternary linear launches with no transient workspace.

#include "ops/linear/q2/q2_dispatch.h"

#include "core/tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

std::size_t q2_gdn_input_proj_workspace_bytes(std::int32_t input_rows, std::int32_t max_tokens) {
    if (input_rows <= 0 || max_tokens <= 0) {
        throw std::invalid_argument("q2 gdn_input_proj: invalid workspace geometry");
    }
    return static_cast<std::size_t>(input_rows) * max_tokens * 2u;
}

namespace detail {

namespace {

Weight slice_rows(const Weight& parent, std::int32_t first_row, std::int32_t rows) {
    if (parent.layout != QuantLayout::RowSplit || parent.k <= 0 ||
        parent.k % parent.group_size != 0) {
        throw std::invalid_argument("q2 gdn_input_proj: parent must be row-split");
    }
    const std::uint64_t groups = static_cast<std::uint64_t>(parent.k / parent.group_size);
    Weight out = parent;
    out.n = rows;
    out.shape[0] = out.padded_shape[0] = rows;
    out.qdata = static_cast<const std::uint8_t*>(parent.qdata) +
                static_cast<std::uint64_t>(first_row) * groups * 32u;
    out.scales = static_cast<const std::uint8_t*>(parent.scales) +
                 static_cast<std::uint64_t>(first_row) * groups * 2u;
    return out;
}

void validate_ternary(const Weight& weight, std::int32_t rows, const char* label) {
    if (weight.qtype != QType::Q2_G128_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 128 || weight.group != 128 ||
        weight.n != rows || weight.k != 5120 || weight.shape[0] != rows ||
        weight.shape[1] != 5120 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 5120 || weight.qdata == nullptr || weight.scales == nullptr ||
        weight.qhigh != nullptr) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

} // namespace

void q2_gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                       Tensor& qkv, Tensor& z, WorkspaceArena& workspace,
                       cudaStream_t stream) {
    constexpr std::int32_t kQkRows = 4096;
    constexpr std::int32_t kValueRows = 6144;

    validate_ternary(qk_weight, kQkRows, "qk weight");
    validate_ternary(value_z_weight, kValueRows + kValueRows, "value/z weight");

    const Weight query = slice_rows(qk_weight, 0, kQkRows / 2);
    const Weight key = slice_rows(qk_weight, kQkRows / 2, kQkRows / 2);
    const Weight value = slice_rows(value_z_weight, 0, kValueRows);
    const Weight z_rows = slice_rows(value_z_weight, kValueRows, kValueRows);
    Tensor transformed = q2_prepare_activation(x, qk_weight, workspace, stream);

    Tensor q_out = qkv.slice(0, 0, kQkRows / 2);
    Tensor k_out = qkv.slice(0, kQkRows / 2, kQkRows / 2);
    q2_linear_launch(transformed, query, q_out, stream, false);
    q2_linear_launch(transformed, key, k_out, stream, false);
    Tensor v_out = qkv.slice(0, kQkRows, kValueRows);
    q2_linear_launch(transformed, value, v_out, stream, false);
    q2_linear_launch(transformed, z_rows, z, stream, false);
}

void q2_gdn_input_proj_single(const Tensor& x, const Weight& parent, Tensor& qkv, Tensor& z,
                              WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr std::int32_t kQkRows = 4096;
    constexpr std::int32_t kValueZRows = 12288;
    if (parent.qtype != QType::Q2_G128_FP16 || parent.n != kQkRows + kValueZRows ||
        parent.k != 5120) {
        throw std::invalid_argument("gdn_input_proj: invalid single-parent q2 weight");
    }
    const Weight qk = slice_rows(parent, 0, kQkRows);
    const Weight value_z = slice_rows(parent, kQkRows, kValueZRows);
    q2_gdn_input_proj(x, qk, value_z, qkv, z, workspace, stream);
}

} // namespace detail
} // namespace ninfer::ops
