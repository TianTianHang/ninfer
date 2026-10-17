"""Exact encoder for Prism's PTQ1_0 128-lane ternary blocks."""

from __future__ import annotations

import torch


_GROUP = 128
_BLOCK_BYTES = 28


def _pack_base3(trits: torch.Tensor, width: int) -> torch.Tensor:
    weights = torch.tensor(
        [3**power for power in range(width - 1, -1, -1)],
        dtype=torch.int64,
        device=trits.device,
    )
    raw = (trits.to(torch.int64) + 1) * weights.view(*([1] * (trits.ndim - 1)), width)
    values = raw.sum(-1)
    # llama.cpp stores ceil(value * 256 / 3**width); the decoder extracts
    # each base-3 digit by repeated multiply-and-wrap modulo 256.
    return ((values * 256 + (3**width - 1)) // (3**width)).to(torch.uint8)


def encode_ptq1(codes: torch.Tensor, scales: torch.Tensor, shape: tuple[int, int]) -> bytes:
    """Encode q2-grid codes 0/1/2 as PTQ1 blocks with exact FP16 scales."""
    rows, k = shape
    if tuple(codes.shape) != (rows, k // _GROUP, _GROUP):
        raise ValueError(f"PTQ1 codes have unexpected shape {tuple(codes.shape)}")
    if tuple(scales.shape) != (rows, k // _GROUP):
        raise ValueError(f"PTQ1 scales have unexpected shape {tuple(scales.shape)}")
    if k <= 0 or k % _GROUP:
        raise ValueError("PTQ1 requires K divisible by 128")
    trits = codes.to(torch.int16) - 1
    first = _pack_base3(trits[..., :80].reshape(rows, -1, 5, 16).transpose(-1, -2), 5)
    second = _pack_base3(trits[..., 80:120].reshape(rows, -1, 5, 8).transpose(-1, -2), 5)
    high = _pack_base3(trits[..., 120:].reshape(rows, -1, 4, 2).transpose(-1, -2), 4)
    scale_bytes = scales.to(torch.float16).contiguous().view(torch.uint8)
    blocks = torch.cat((first, second, high, scale_bytes.reshape(rows, -1, 2)), dim=-1)
    return blocks.contiguous().cpu().numpy().tobytes()


__all__ = ["encode_ptq1"]
