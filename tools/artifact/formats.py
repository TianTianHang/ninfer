"""Closed registry of persistent NInfer tensor numeric formats."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from types import MappingProxyType
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class DirectFormat:
    """One fixed-width word per logical tensor element."""

    name: str
    word_bytes: int


@dataclass(frozen=True, slots=True)
class QuantFormat:
    """Grouped codes with one binary16 multiplier per group.

    ``qmin``/``qmax`` bound the representable weight coefficients. Two code
    conventions exist, selected by ``zero_value_code``:

    * ``None`` — the packed code is a two's-complement integer and is itself the
      weight coefficient, so the represented value is ``code * scale``.
    * an integer — the packed code is an unsigned grid index in
      ``0 .. 2**bits - 1`` and the represented value is
      ``(code - zero_value_code) * scale``.
    """

    name: str
    bits: int
    group_size: int
    qmin: int
    qmax: int
    zero_value_code: int | None = None

    def __post_init__(self) -> None:
        if self.zero_value_code is not None and not (
            0 <= self.zero_value_code < 1 << self.bits
        ):
            raise ValueError(
                f"{self.name}: zero_value_code must be a {self.bits}-bit grid index"
            )


@dataclass(frozen=True, slots=True)
class PackedTernaryFormat:
    """Prism PTQ1 base-3 ternary blocks with one FP16 scale per 128 lanes."""

    name: str
    group_size: int
    block_bytes: int


@dataclass(frozen=True, slots=True)
class Nvfp4Format:
    """E2M1 weights with one E4M3FN scale word per K-axis group."""

    name: str
    group_size: int


@dataclass(frozen=True, slots=True)
class Fp8RowFormat:
    """E4M3FN weights with one BF16 multiplier per logical row."""

    name: str


NumericFormat: TypeAlias = DirectFormat | QuantFormat | PackedTernaryFormat | Nvfp4Format | Fp8RowFormat


BF16 = DirectFormat("bf16", 2)
FP32 = DirectFormat("fp32", 4)
INT32 = DirectFormat("int32", 4)

Q4_G64_FP16 = QuantFormat("q4_g64_fp16", 4, 64, -8, 7)
Q5_G64_FP16 = QuantFormat("q5_g64_fp16", 5, 64, -16, 15)
Q6_G64_FP16 = QuantFormat("q6_g64_fp16", 6, 64, -32, 31)
Q8_G32_FP16 = QuantFormat("q8_g32_fp16", 8, 32, -127, 127)
# Ternary codes on an unsigned grid: code c in {0,1,2,3} represents (c - 1) * scale,
# i.e. the coefficients {-1, 0, +1, +2}. This is exactly the ggml PQ2_0 codebook, so
# an existing PQ2_0 payload can be imported without touching codes or scales.
Q2_G128_FP16 = QuantFormat("q2_g128_fp16", 2, 128, -1, 2, zero_value_code=1)
PTQ1_G128_FP16 = PackedTernaryFormat("ptq1_g128_fp16", 128, 28)
NVFP4 = Nvfp4Format("nvfp4", 16)
FP8_E4M3FN_ROW_BF16 = Fp8RowFormat("fp8_e4m3fn_row_bf16")


DIRECT_FORMATS = MappingProxyType({item.name: item for item in (BF16, FP32, INT32)})
QUANT_FORMATS = MappingProxyType(
    {
        item.name: item
        for item in (Q2_G128_FP16, Q4_G64_FP16, Q5_G64_FP16, Q6_G64_FP16, Q8_G32_FP16)
    }
)
PACKED_TERNARY_FORMATS = MappingProxyType({PTQ1_G128_FP16.name: PTQ1_G128_FP16})
NVFP4_FORMATS = MappingProxyType({NVFP4.name: NVFP4})
FP8_ROW_FORMATS = MappingProxyType({FP8_E4M3FN_ROW_BF16.name: FP8_E4M3FN_ROW_BF16})
NUMERIC_FORMATS = MappingProxyType(
    {**DIRECT_FORMATS, **QUANT_FORMATS, **PACKED_TERNARY_FORMATS, **NVFP4_FORMATS, **FP8_ROW_FORMATS}
)


_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def decode_e2m1_word(word: int) -> float:
    """Decode one exact four-bit E2M1 word, including signed zero."""

    if type(word) is not int or not 0 <= word <= 0xF:
        raise ValueError("E2M1 word must be an integer in [0, 15]")
    magnitude = _E2M1_MAGNITUDES[word & 0x7]
    return math.copysign(magnitude, -1.0 if word & 0x8 else 1.0)


def decode_e4m3fn_word(word: int) -> float:
    """Decode one exact eight-bit E4M3FN word."""

    if type(word) is not int or not 0 <= word <= 0xFF:
        raise ValueError("E4M3FN word must be an integer in [0, 255]")
    sign = -1.0 if word & 0x80 else 1.0
    exponent = (word >> 3) & 0xF
    fraction = word & 0x7
    if exponent == 0:
        if fraction == 0:
            return math.copysign(0.0, sign)
        return sign * fraction * (2.0**-9)
    if exponent == 0xF and fraction == 0x7:
        return math.copysign(math.nan, sign)
    return sign * (1.0 + fraction / 8.0) * (2.0 ** (exponent - 7))


def valid_nvfp4_scale_word(word: int) -> bool:
    """Return whether *word* is an admitted nonnegative finite E4M3FN scale."""

    return type(word) is int and 0 <= word <= 0xFF and word & 0x80 == 0 and word != 0x7F


def valid_fp8_weight_word(word: int) -> bool:
    """Return whether *word* is a finite E4M3FN weight code."""

    return type(word) is int and 0 <= word <= 0xFF and (word & 0x7F) != 0x7F


def valid_fp8_row_scale_word(word: int) -> bool:
    """Return whether *word* is a nonnegative finite BF16 multiplier."""

    if type(word) is not int or not 0 <= word <= 0xFFFF or word & 0x8000:
        return False
    value = struct.unpack("<f", struct.pack("<I", word << 16))[0]
    return math.isfinite(value)


def valid_positive_fp32_word(word: int) -> bool:
    """Return whether an IEEE binary32 word represents a finite positive value."""

    if type(word) is not int or not 0 <= word <= 0xFFFFFFFF:
        return False
    value = struct.unpack("<f", struct.pack("<I", word))[0]
    return math.isfinite(value) and value > 0.0


def get_format(name: str) -> NumericFormat:
    """Return the registered format named *name*."""

    try:
        return NUMERIC_FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown numeric format: {name!r}") from None


__all__ = [
    "BF16",
    "FP32",
    "INT32",
    "Q2_G128_FP16",
    "PTQ1_G128_FP16",
    "Q4_G64_FP16",
    "Q5_G64_FP16",
    "Q6_G64_FP16",
    "Q8_G32_FP16",
    "NVFP4",
    "FP8_E4M3FN_ROW_BF16",
    "DIRECT_FORMATS",
    "QUANT_FORMATS",
    "PACKED_TERNARY_FORMATS",
    "NVFP4_FORMATS",
    "FP8_ROW_FORMATS",
    "NUMERIC_FORMATS",
    "DirectFormat",
    "QuantFormat",
    "PackedTernaryFormat",
    "Nvfp4Format",
    "Fp8RowFormat",
    "NumericFormat",
    "get_format",
    "decode_e2m1_word",
    "decode_e4m3fn_word",
    "valid_fp8_row_scale_word",
    "valid_fp8_weight_word",
    "valid_nvfp4_scale_word",
    "valid_positive_fp32_word",
]
