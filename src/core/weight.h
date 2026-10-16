#pragma once

#include "core/dtype.h"

#include <cstdint>

namespace ninfer {

enum class QType : std::uint16_t {
    Q4_G64_FP16         = 0,
    Q5_G64_FP16         = 1,
    Q6_G64_FP16         = 2,
    Q8_G32_FP16         = 3,
    BF16                = 4,
    FP32                = 5,
    INT32               = 6,
    NVFP4               = 7,
    FP8_E4M3FN_ROW_BF16 = 8,
    // Ternary weights on an unsigned 2-bit grid: code c in [0,3] represents
    // (c - 1) * scale, i.e. the coefficients {-1, 0, +1, +2}. This is the ggml
    // PQ2_0 codebook, so an existing PQ2_0 payload is imported without rewriting
    // a single code or scale.
    Q2_G128_FP16        = 9,
};

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
};

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4_G64_FP16;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;

    // Optional model-provided activation-side Hadamard metadata. The model owns
    // and materializes the sign vector; Weight only borrows its device address.
    const std::int32_t* hadamard_signs = nullptr;
    std::int32_t hadamard_block_size = 0;
    std::int32_t hadamard_perm_hd = 0;
    std::int32_t hadamard_perm_nk = 0;
    std::int32_t hadamard_perm_rep = 0;
    bool inverse_hadamard = false;
};

} // namespace ninfer
