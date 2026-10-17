// Native ternary A16 contraction (Q2_G128_FP16 / PTQ1_G128_FP16).
//
// Both formats describe one 128-lane quantization group with one binary16 scale:
//
//   Q2   32 code bytes, lane j uses the low-first two-bit slot of byte j/4 and
//        the unsigned grid value (code - 1) in {-1,0,+1,+2};
//   PTQ1 28 bytes: 26 canonical base-3 digit bytes covering 128 lanes plus the
//        binary16 scale in the last two bytes. Digit byte b < 16 carries lanes
//        b + 16*d, byte 16 <= b < 24 carries lanes 80 + (b-16) + 8*d, and byte
//        24 <= b < 26 carries lanes 120 + (b-24) + 2*d.
//
// The routes exploit that the coefficient grid is ternary (Q2 additionally has
// +2): a lane that owns a code byte decodes the whole group slice once and reuses
// it over several output rows and token columns. No runtime weight repacking or
// persistent workspace is involved.
//
//   T <= 8  warp contraction. Lane j owns code byte j of the current group and
//           accumulates kRows rows over the whole token extent, so weight bytes
//           are read and decoded exactly once per row group.
//   T > 8   tensor-core block contraction. A CTA decodes one weight group for 64
//           rows and stages the matching activation columns for 64 tokens as
//           shared bf16 tiles, then contracts them with BF16 mma.sync under FP32
//           accumulation. Each weight byte is decoded once per CTA and reused
//           across every staged token.
//
// The per-group scale multiplies the completed group dot product, so the scale
// is applied exactly once per group instead of once per weight.

#include "ops/linear/q2/ternary_a16.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kGroup = 128;

enum class TernaryFormat : std::uint8_t { Q2, Ptq1 };

template <TernaryFormat F>
struct Traits;
template <>
struct Traits<TernaryFormat::Q2> {
    static constexpr int kBytesPerGroup = 32;
    static constexpr int kMaxDigits     = 4;
    static constexpr bool kScalePlane   = true;
};
template <>
struct Traits<TernaryFormat::Ptq1> {
    static constexpr int kBytesPerGroup = 28;
    static constexpr int kMaxDigits     = 5;
    static constexpr bool kScalePlane   = false;
};

// ---------------------------------------------------------------------------
// Warp contraction
// ---------------------------------------------------------------------------

constexpr int kGemvThreads = 256;
constexpr int kGemvWarps   = kGemvThreads / 32;

struct LaneMap {
    int base;     // first lane element of the owned code byte
    int stride;   // element distance between the byte's digits
    int digits;   // digits carried by the owned byte
    bool active;  // contributes to the reduction
};

template <TernaryFormat F>
__device__ __forceinline__ LaneMap lane_map(int lane) {
    LaneMap map{};
    if constexpr (F == TernaryFormat::Ptq1) {
        if (lane < 16) {
            map.base   = lane;
            map.stride = 16;
            map.digits = 5;
        } else if (lane < 24) {
            map.base   = 80 + (lane - 16);
            map.stride = 8;
            map.digits = 5;
        } else {
            map.base   = 120 + (lane - 24);
            map.stride = 2;
            map.digits = 4;
        }
        map.active = lane < 26;
    } else {
        map.base   = lane * 4;
        map.stride = 1;
        map.digits = 4;
        map.active = true;
    }
    return map;
}

template <TernaryFormat F, int kTokens, int kRows, bool kExactTokens, bool kAligned>
__global__ void ternary_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                    const std::uint8_t* __restrict__ codes,
                                    const std::uint8_t* __restrict__ scales,
                                    __nv_bfloat16* __restrict__ out, int n, int k, int groups,
                                    int x_token_stride, int out_token_stride, int t,
                                    bool accumulate) {
    constexpr int kBytes = Traits<F>::kBytesPerGroup;
    constexpr int kMaxDigits = Traits<F>::kMaxDigits;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row_base = (static_cast<int>(blockIdx.x) * kGemvWarps + warp) * kRows;
    const LaneMap map = lane_map<F>(lane);
    const std::int64_t row_stride = static_cast<std::int64_t>(groups) * kBytes;

    // One code byte per lane. PTQ1 lanes 28..31 are outside the 28-byte group.
    const bool lane_loads = lane < kBytes;

    bool row_ok[kRows];
    bool row_valid[kRows];
    const std::uint8_t* row_codes[kRows];
    const std::uint8_t* row_scales[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const int row     = row_base + r;
        row_ok[r]         = row < n;
        const int clamped = row_ok[r] ? row : n - 1;
        row_codes[r]      = codes + static_cast<std::int64_t>(clamped) * row_stride;
        row_scales[r]     = scales == nullptr
                                ? nullptr
                                : scales + static_cast<std::int64_t>(clamped) * groups * 2;
        row_valid[r]      = row_ok[r] && map.active;
    }

    float acc[kRows][kTokens];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
#pragma unroll
        for (int u = 0; u < kTokens; ++u) { acc[r][u] = 0.0f; }
    }

    for (int g = 0; g < groups; ++g) {
        std::uint8_t byte[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            byte[r] = (row_ok[r] && lane_loads) ? row_codes[r][lane] : static_cast<std::uint8_t>(0);
        }
        float scale[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            float value = 0.0f;
            if (row_valid[r]) {
                if constexpr (Traits<F>::kScalePlane) {
                    value = __half2float(*reinterpret_cast<const __half*>(row_scales[r]));
                } else {
                    value = __half2float(
                        *reinterpret_cast<const __half*>(row_codes[r] + kBytes - 2));
                }
            }
            scale[r] = value;
        }

        float sum[kRows][kTokens];
        float xsum[kTokens];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int u = 0; u < kTokens; ++u) { sum[r][u] = 0.0f; }
        }
#pragma unroll
        for (int u = 0; u < kTokens; ++u) { xsum[u] = 0.0f; }

#pragma unroll
        for (int a = 0; a < kMaxDigits; ++a) {
            const bool has_digit = a < map.digits;
            float xa[kTokens];
#pragma unroll
            for (int u = 0; u < kTokens; ++u) {
                float value = 0.0f;
                const bool token_ok = kExactTokens || u < t;
                if (has_digit && token_ok && map.active) {
                    const int element = map.base + a * map.stride;
                    if (kAligned || g * kGroup + element < k) {
                        value = __bfloat162float(
                            x[static_cast<std::int64_t>(u) * x_token_stride + g * kGroup +
                              element]);
                    }
                }
                xa[u] = value;
                xsum[u] += value;
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                int code;
                if constexpr (F == TernaryFormat::Ptq1) {
                    const std::uint32_t product = static_cast<std::uint32_t>(byte[r]) * 3U;
                    code                        = static_cast<int>(product >> 8);
                    byte[r]                     = static_cast<std::uint8_t>(product & 0xffU);
                } else {
                    code = (byte[r] >> (2 * a)) & 3;
                }
                const float value = static_cast<float>(code);
#pragma unroll
                for (int u = 0; u < kTokens; ++u) {
                    sum[r][u] = fmaf(value, xa[u], sum[r][u]);
                }
            }
        }

#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int u = 0; u < kTokens; ++u) {
                acc[r][u] = fmaf(sum[r][u] - xsum[u], scale[r], acc[r][u]);
            }
            row_codes[r] += kBytes;
            if constexpr (Traits<F>::kScalePlane) { row_scales[r] += 2; }
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int u = 0; u < kTokens; ++u) {
                acc[r][u] += __shfl_down_sync(0xffffffffU, acc[r][u], offset);
            }
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            if (!row_ok[r]) { continue; }
#pragma unroll
            for (int u = 0; u < kTokens; ++u) {
                if (!kExactTokens && u >= t) { continue; }
                const std::int64_t at = static_cast<std::int64_t>(u) * out_token_stride +
                                        (row_base + r);
                const float value = accumulate ? acc[r][u] + __bfloat162float(out[at]) : acc[r][u];
                out[at] = __float2bfloat16(value);
            }
        }
    }
}

// T=1 fast path: every warp covers a fixed number of complete rows, so no row
// bounds, token bounds, or stride checks remain in the group loop. Byte, scale,
// and activation cursors are advanced by constants so the hot loop carries only
// the base-3 decode, the FMA, and five loads.
template <TernaryFormat F, int kRows>
__global__ void ternary_gemv1_kernel(const __nv_bfloat16* __restrict__ x,
                                     const std::uint8_t* __restrict__ codes,
                                     const std::uint8_t* __restrict__ scales,
                                     __nv_bfloat16* __restrict__ out, int groups,
                                     bool accumulate) {
    constexpr int kBytes    = Traits<F>::kBytesPerGroup;
    constexpr int kMaxDigit = Traits<F>::kMaxDigits;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row0 = (static_cast<int>(blockIdx.x) * kGemvWarps + warp) * kRows;
    const LaneMap map = lane_map<F>(lane);
    const bool lane_live = map.active;

    const std::uint8_t* code_cursor[kRows];
    const std::uint8_t* scale_cursor[kRows];
    float row_scale[kRows];
    const std::int64_t row_bytes = static_cast<std::int64_t>(groups) * kBytes;
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const int row = row0 + r;
        code_cursor[r] = codes + static_cast<std::int64_t>(row) * row_bytes + lane;
        if constexpr (Traits<F>::kScalePlane) {
            scale_cursor[r] = scales + static_cast<std::int64_t>(row) * groups * 2;
        } else {
            scale_cursor[r] = codes + static_cast<std::int64_t>(row) * row_bytes;
        }
    }

    const __nv_bfloat16* activation = x + map.base;
    int digit_offset[kMaxDigit];
#pragma unroll
    for (int a = 0; a < kMaxDigit; ++a) { digit_offset[a] = a * map.stride; }

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) { acc[r] = 0.0f; }

    for (int g = 0; g < groups; ++g) {
        std::uint8_t byte[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) { byte[r] = code_cursor[r][0]; }
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            float scale = 0.0f;
            if (lane_live) {
                if constexpr (Traits<F>::kScalePlane) {
                    scale = __half2float(*reinterpret_cast<const __half*>(scale_cursor[r]));
                } else {
                    scale = __half2float(
                        *reinterpret_cast<const __half*>(scale_cursor[r] + kBytes - 2));
                }
            }
            row_scale[r] = scale;
        }

        float value[kMaxDigit];
#pragma unroll
        for (int a = 0; a < kMaxDigit; ++a) {
            const bool live = a < map.digits && lane_live;
            value[a] = live ? __bfloat162float(activation[digit_offset[a]]) : 0.0f;
        }
        float xsum = value[0];
#pragma unroll
        for (int a = 1; a < kMaxDigit; ++a) { xsum += value[a]; }

        float sum[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) { sum[r] = 0.0f; }
#pragma unroll
        for (int a = 0; a < kMaxDigit; ++a) {
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                int code;
                if constexpr (F == TernaryFormat::Ptq1) {
                    const std::uint32_t product = static_cast<std::uint32_t>(byte[r]) * 3U;
                    code                        = static_cast<int>(product >> 8);
                    byte[r]                     = static_cast<std::uint8_t>(product & 0xffU);
                } else {
                    code = (byte[r] >> (2 * a)) & 3;
                }
                sum[r] = fmaf(static_cast<float>(code), value[a], sum[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            acc[r] = fmaf(sum[r] - xsum, row_scale[r], acc[r]);
            code_cursor[r] += kBytes;
            if constexpr (Traits<F>::kScalePlane) { scale_cursor[r] += 2; }
            else { scale_cursor[r] += kBytes; }
        }
        activation += kGroup;
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) { acc[r] += __shfl_down_sync(0xffffffffU, acc[r], offset); }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int row = row0 + r;
            const float value = accumulate ? acc[r] + __bfloat162float(out[row]) : acc[r];
            out[row]          = __float2bfloat16(value);
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor-core block contraction
// ---------------------------------------------------------------------------
//
// The arithmetic is unchanged: ternary weights are decoded to exact bf16
// coefficients in shared memory and multiplied with the bf16 activations under
// FP32 accumulation. Only the contraction instruction changes: instead of
// scalar FFMA per output element, each warp issues BF16 mma.sync matrix
// instructions and applies the per-group scale to the completed group product.

constexpr int kMmaRows       = 96;
constexpr int kMmaTokens     = 64;
constexpr int kMmaStride     = 136;  // 128 lanes + 8 bf16 of row padding
constexpr int kMmaThreads    = 768;
constexpr int kMmaWarpRows   = 16;
constexpr int kMmaWarpTokens = 16;
constexpr int kMmaNTiles     = kMmaWarpTokens / 8;
constexpr int kMmaWarpCols   = kMmaTokens / kMmaWarpTokens;
constexpr int kMmaLoadChunks = kMmaThreads / kMmaTokens;
constexpr int kMmaLoadElems  = kGroup / kMmaLoadChunks;

__device__ __forceinline__ void ldmatrix_x4(std::uint32_t (&reg)[4], const void* address) {
    const std::uint32_t shared = static_cast<std::uint32_t>(__cvta_generic_to_shared(address));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
                 : "r"(shared));
}

__device__ __forceinline__ void ldmatrix_x2(std::uint32_t (&reg)[2], const void* address) {
    const std::uint32_t shared = static_cast<std::uint32_t>(__cvta_generic_to_shared(address));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(reg[0]), "=r"(reg[1])
                 : "r"(shared));
}

__device__ __forceinline__ void mma_16x8x16(float (&d)[4], const std::uint32_t (&a)[4],
                                            const std::uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <TernaryFormat F, bool kAligned>
__global__ void ternary_gemm_mma_kernel(const __nv_bfloat16* __restrict__ x,
                                        const std::uint8_t* __restrict__ codes,
                                        const std::uint8_t* __restrict__ scales,
                                        __nv_bfloat16* __restrict__ out, int n, int k, int groups,
                                        int x_token_stride, int out_token_stride, int t,
                                        bool accumulate) {
    constexpr int kBytes = Traits<F>::kBytesPerGroup;
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int row0   = static_cast<int>(blockIdx.x) * kMmaRows;
    const int token0 = static_cast<int>(blockIdx.y) * kMmaTokens;
    const int warp_row = (warp / kMmaWarpCols) * kMmaWarpRows;
    const int warp_col = (warp % kMmaWarpCols) * kMmaWarpTokens;
    const std::int64_t row_bytes = static_cast<std::int64_t>(groups) * kBytes;

    __shared__ __align__(16) __nv_bfloat16 a_tile[kMmaRows][kMmaStride];
    __shared__ __align__(16) __nv_bfloat16 b_tile[kMmaTokens][kMmaStride];
    __shared__ float tile_scale[kMmaRows];

    float acc[kMmaNTiles][4];
#pragma unroll
    for (int j = 0; j < kMmaNTiles; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) { acc[j][e] = 0.0f; }
    }

    for (int g = 0; g < groups; ++g) {
        __syncthreads();

        // Stage the decoded weight group as [row][lane]; the decode is identical
        // to the scalar route, only the shared-memory order changes.
        if constexpr (F == TernaryFormat::Ptq1) {
            for (int index = tid; index < kMmaRows * 26; index += kMmaThreads) {
                const int rl   = index / 26;
                const int byte = index - rl * 26;
                const int row  = row0 + rl;
                std::uint32_t value = row < n ? codes[static_cast<std::int64_t>(row) * row_bytes +
                                                      g * kBytes + byte]
                                              : 0U;
                int element;
                int stride;
                int digits;
                if (byte < 16) {
                    element = byte;
                    stride  = 16;
                    digits  = 5;
                } else if (byte < 24) {
                    element = 80 + (byte - 16);
                    stride  = 8;
                    digits  = 5;
                } else {
                    element = 120 + (byte - 24);
                    stride  = 2;
                    digits  = 4;
                }
                for (int a = 0; a < 5; ++a) {
                    if (a >= digits) { break; }
                    const std::uint32_t product = value * 3U;
                    const int digit             = static_cast<int>(product >> 8);
                    value                       = product & 0xffU;
                    a_tile[rl][element + a * stride] =
                        __float2bfloat16(static_cast<float>(digit) - 1.0f);
                }
            }
            for (int rl = tid; rl < kMmaRows; rl += kMmaThreads) {
                const int row = row0 + rl;
                tile_scale[rl] =
                    row < n ? __half2float(*reinterpret_cast<const __half*>(
                                  codes + static_cast<std::int64_t>(row) * row_bytes + g * kBytes +
                                  kBytes - 2))
                            : 0.0f;
            }
        } else {
            for (int index = tid; index < kMmaRows * 32; index += kMmaThreads) {
                const int rl   = index / 32;
                const int byte = index - rl * 32;
                const int row  = row0 + rl;
                const std::uint32_t value =
                    row < n
                        ? codes[static_cast<std::int64_t>(row) * row_bytes + g * kBytes + byte]
                        : 0U;
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    const int code = static_cast<int>((value >> (2 * a)) & 3U);
                    a_tile[rl][4 * byte + a] = __float2bfloat16(static_cast<float>(code) - 1.0f);
                }
            }
            for (int rl = tid; rl < kMmaRows; rl += kMmaThreads) {
                const int row = row0 + rl;
                tile_scale[rl] =
                    row < n ? __half2float(*reinterpret_cast<const __half*>(
                                  scales + (static_cast<std::int64_t>(row) * groups + g) * 2))
                            : 0.0f;
            }
        }

        // Stage the activation group as [token][lane] with vector copies. Each
        // unit is eight consecutive lanes; the loop adapts to the warp count.
        for (int unit = tid; unit < kMmaTokens * (kGroup / 8); unit += kMmaThreads) {
            const int token_local = unit / (kGroup / 8);
            const int kbase       = (unit % (kGroup / 8)) * 8;
            const int token       = token0 + token_local;
            __nv_bfloat16* dst    = &b_tile[token_local][kbase];
            if (token < t) {
                const __nv_bfloat16* src =
                    x + static_cast<std::int64_t>(token) * x_token_stride + g * kGroup + kbase;
                if (kAligned || g * kGroup + kbase + 7 < k) {
                    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
                } else {
                    for (int i = 0; i < 8; ++i) {
                        dst[i] = (g * kGroup + kbase + i < k) ? src[i] : __float2bfloat16(0.0f);
                    }
                }
            } else {
                *reinterpret_cast<uint4*>(dst) = uint4{0U, 0U, 0U, 0U};
            }
        }

        __syncthreads();

        float group_acc[kMmaNTiles][4];
#pragma unroll
        for (int j = 0; j < kMmaNTiles; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { group_acc[j][e] = 0.0f; }
        }

#pragma unroll
        for (int k0 = 0; k0 < kGroup; k0 += 16) {
            std::uint32_t a[4];
            ldmatrix_x4(a, &a_tile[warp_row + (lane & 15)][k0 + 8 * (lane >> 4)]);
            std::uint32_t b[kMmaNTiles][2];
#pragma unroll
            for (int j = 0; j < kMmaNTiles; ++j) {
                ldmatrix_x2(b[j], &b_tile[warp_col + 8 * j + (lane & 7)][k0 + 8 * ((lane >> 3) & 1)]);
            }
#pragma unroll
            for (int j = 0; j < kMmaNTiles; ++j) { mma_16x8x16(group_acc[j], a, b[j]); }
        }

        const int scale_row0 = warp_row + (lane >> 2);
        const int scale_row1 = scale_row0 + 8;
        const float s0       = tile_scale[scale_row0];
        const float s1       = tile_scale[scale_row1];
#pragma unroll
        for (int j = 0; j < kMmaNTiles; ++j) {
            acc[j][0] = fmaf(group_acc[j][0], s0, acc[j][0]);
            acc[j][1] = fmaf(group_acc[j][1], s0, acc[j][1]);
            acc[j][2] = fmaf(group_acc[j][2], s1, acc[j][2]);
            acc[j][3] = fmaf(group_acc[j][3], s1, acc[j][3]);
        }
    }

    const int out_row0 = row0 + warp_row + (lane >> 2);
    const int out_row1 = out_row0 + 8;
#pragma unroll
    for (int j = 0; j < kMmaNTiles; ++j) {
        const int col = token0 + warp_col + 8 * j + 2 * (lane & 3);
        if (out_row0 < n && col < t) {
            const std::int64_t at = static_cast<std::int64_t>(col) * out_token_stride + out_row0;
            const float value = accumulate ? acc[j][0] + __bfloat162float(out[at]) : acc[j][0];
            out[at]           = __float2bfloat16(value);
        }
        if (out_row0 < n && col + 1 < t) {
            const std::int64_t at = static_cast<std::int64_t>(col + 1) * out_token_stride + out_row0;
            const float value = accumulate ? acc[j][1] + __bfloat162float(out[at]) : acc[j][1];
            out[at]           = __float2bfloat16(value);
        }
        if (out_row1 < n && col < t) {
            const std::int64_t at = static_cast<std::int64_t>(col) * out_token_stride + out_row1;
            const float value = accumulate ? acc[j][2] + __bfloat162float(out[at]) : acc[j][2];
            out[at]           = __float2bfloat16(value);
        }
        if (out_row1 < n && col + 1 < t) {
            const std::int64_t at = static_cast<std::int64_t>(col + 1) * out_token_stride + out_row1;
            const float value = accumulate ? acc[j][3] + __bfloat162float(out[at]) : acc[j][3];
            out[at]           = __float2bfloat16(value);
        }
    }
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

template <TernaryFormat F, int kRows>
void launch_gemv1(const __nv_bfloat16* x, const std::uint8_t* codes,
                  const std::uint8_t* scales, __nv_bfloat16* out, int n, int k, int groups,
                  int out_stride, bool accumulate, cudaStream_t stream) {
    constexpr int kRowsPerBlock = kRows * kGemvWarps;
    const int blocks            = n / kRowsPerBlock;
    (void)k;
    (void)out_stride;
    ternary_gemv1_kernel<F, kRows><<<blocks, kGemvThreads, 0, stream>>>(
        x, codes, scales, out, groups, accumulate);
    CUDA_CHECK(cudaGetLastError());
}

template <TernaryFormat F, int kTokens, int kRows, bool kExactTokens, bool kAligned>
void launch_gemv(const __nv_bfloat16* x, const std::uint8_t* codes, const std::uint8_t* scales,
                 __nv_bfloat16* out, int n, int k, int groups, int x_stride, int out_stride,
                 int t, bool accumulate, cudaStream_t stream) {
    constexpr int kRowsPerBlock = kRows * kGemvWarps;
    const int blocks            = (n + kRowsPerBlock - 1) / kRowsPerBlock;
    ternary_gemv_kernel<F, kTokens, kRows, kExactTokens, kAligned>
        <<<blocks, kGemvThreads, 0, stream>>>(x, codes, scales, out, n, k, groups, x_stride,
                                              out_stride, t, accumulate);
    CUDA_CHECK(cudaGetLastError());
}

// A warp that covers kRows rows exposes n/kRows warps to the scheduler and
// occupies one block per kRows*8 rows. A row group only pays off when the grid
// still spans several device waves; below that, the wave tail costs more than
// the per-group activation and scale loads that a larger group amortizes.
constexpr int kRowsPerWarp(int n) { return n >= 6144 ? 4 : 1; }

template <TernaryFormat F, bool kAligned>
void dispatch_gemv(const __nv_bfloat16* x, const std::uint8_t* codes,
                   const std::uint8_t* scales, __nv_bfloat16* out, int n, int k, int groups,
                   int x_stride, int out_stride, int t, bool accumulate,
                   cudaStream_t stream) {
    const int rows = kRowsPerWarp(n);
    if (t == 1) {
        if (kAligned && n % (rows * kGemvWarps) == 0) {
            if (rows == 1) {
                launch_gemv1<F, 1>(x, codes, scales, out, n, k, groups, out_stride, accumulate,
                                   stream);
            } else {
                launch_gemv1<F, 4>(x, codes, scales, out, n, k, groups, out_stride, accumulate,
                                   stream);
            }
            return;
        }
        if (rows == 1) {
            launch_gemv<F, 1, 1, true, kAligned>(x, codes, scales, out, n, k, groups, x_stride,
                                                 out_stride, t, accumulate, stream);
        } else {
            launch_gemv<F, 1, 4, true, kAligned>(x, codes, scales, out, n, k, groups, x_stride,
                                                 out_stride, t, accumulate, stream);
        }
    } else if (t == 2) {
        if (rows == 1) {
            launch_gemv<F, 2, 1, true, kAligned>(x, codes, scales, out, n, k, groups, x_stride,
                                                 out_stride, t, accumulate, stream);
        } else {
            launch_gemv<F, 2, 4, true, kAligned>(x, codes, scales, out, n, k, groups, x_stride,
                                                 out_stride, t, accumulate, stream);
        }
    } else if (t == 4) {
        launch_gemv<F, 4, 4, true, kAligned>(x, codes, scales, out, n, k, groups, x_stride,
                                             out_stride, t, accumulate, stream);
    } else if (t == 8) {
        launch_gemv<F, 8, 2, true, kAligned>(x, codes, scales, out, n, k, groups, x_stride,
                                             out_stride, t, accumulate, stream);
    } else if (t < 4) {
        launch_gemv<F, 4, 4, false, kAligned>(x, codes, scales, out, n, k, groups, x_stride,
                                              out_stride, t, accumulate, stream);
    } else {
        launch_gemv<F, 8, 2, false, kAligned>(x, codes, scales, out, n, k, groups, x_stride,
                                              out_stride, t, accumulate, stream);
    }
}

template <TernaryFormat F>
void dispatch_gemm(const __nv_bfloat16* x, const std::uint8_t* codes,
                   const std::uint8_t* scales, __nv_bfloat16* out, int n, int k, int groups,
                   int x_stride, int out_stride, int t, bool accumulate, bool aligned,
                   cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>((n + kMmaRows - 1) / kMmaRows),
                    static_cast<unsigned>((t + kMmaTokens - 1) / kMmaTokens));
    if (aligned) {
        ternary_gemm_mma_kernel<F, true><<<grid, kMmaThreads, 0, stream>>>(
            x, codes, scales, out, n, k, groups, x_stride, out_stride, t, accumulate);
    } else {
        ternary_gemm_mma_kernel<F, false><<<grid, kMmaThreads, 0, stream>>>(
            x, codes, scales, out, n, k, groups, x_stride, out_stride, t, accumulate);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <TernaryFormat F>
void launch_format(const Tensor& x, const Weight& weight, Tensor& out, int t, cudaStream_t stream,
                   bool accumulate) {
    const int n      = weight.n;
    const int k      = weight.k;
    const int groups = weight.padded_shape[1] / kGroup;
    const bool aligned = (k % kGroup) == 0;
    const auto* xp     = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes  = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales);
    auto* op           = static_cast<__nv_bfloat16*>(out.data);
    if (x.nb[1] % static_cast<std::int64_t>(sizeof(__nv_bfloat16)) != 0 ||
        out.nb[1] % static_cast<std::int64_t>(sizeof(__nv_bfloat16)) != 0) {
        throw std::invalid_argument("ternary linear: token stride is not a whole bf16 count");
    }
    const int x_stride   = static_cast<int>(x.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
    const int out_stride = static_cast<int>(out.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));

    if (t <= 8) {
        if (aligned) {
            dispatch_gemv<F, true>(xp, codes, scales, op, n, k, groups, x_stride, out_stride, t,
                                   accumulate, stream);
        } else {
            dispatch_gemv<F, false>(xp, codes, scales, op, n, k, groups, x_stride, out_stride, t,
                                    accumulate, stream);
        }
        return;
    }
    dispatch_gemm<F>(xp, codes, scales, op, n, k, groups, x_stride, out_stride, t, accumulate,
                     aligned, stream);
}

} // namespace

void ternary_a16_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
                        bool accumulate) {
    const bool q2   = weight.qtype == QType::Q2_G128_FP16;
    const bool ptq1 = weight.qtype == QType::PTQ1_G128_FP16;
    if (!q2 && !ptq1) {
        throw std::invalid_argument("ternary linear: weight must be q2 or ptq1 row-split");
    }
    if (weight.layout != QuantLayout::RowSplit || weight.scale_dtype != DType::FP16 ||
        weight.ndim != 2 || weight.group_size != kGroup || weight.group != kGroup ||
        weight.n <= 0 || weight.k <= 0 || weight.shape[0] != weight.n ||
        weight.shape[1] != weight.k || weight.padded_shape[0] != weight.n ||
        weight.padded_shape[1] < weight.k || weight.padded_shape[1] % kGroup != 0 ||
        weight.qdata == nullptr || weight.qhigh != nullptr ||
        (q2 && weight.scales == nullptr) || (ptq1 && weight.scales != nullptr)) {
        throw std::invalid_argument("ternary linear: invalid row-split weight");
    }
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || x.data == nullptr ||
        out.data == nullptr || x.ne[0] != weight.k || out.ne[0] != weight.n ||
        x.ne[1] != out.ne[1] || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[2] != 1 ||
        out.ne[3] != 1) {
        throw std::invalid_argument("ternary linear: operand geometry disagrees with the weight");
    }
    const int t = x.ne[1];
    if (t <= 0) { throw std::invalid_argument("ternary linear: T must be positive"); }

    if (q2) {
        launch_format<TernaryFormat::Q2>(x, weight, out, t, stream, accumulate);
    } else {
        launch_format<TernaryFormat::Ptq1>(x, weight, out, t, stream, accumulate);
    }
}

} // namespace ninfer::ops::detail
