"""GGUF checkpoint access for the converter: metadata, tensors, and exact code import.

Only the encodings the Ternary-Bonsai checkpoint uses are decoded. `read_encoded`
exposes the stored PQ2_0 codes and fp16 group scales unchanged, so a ternary target
imports the checkpoint without rewriting a code or a scale (see
`q2_g128_fp16` in `tools.artifact.formats`).

Type ids follow ggml (`ggml.h`):
    0   F32     32-bit float
    30  BF16    bfloat16                     (declared; see `half` below)
    142 PQ2_0   ternaries, group 128, 2-bit slots, one fp16 scale per group
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Iterator, Literal

import torch

from tools.artifact.file_io import discard_cached_pages

from .logical import EncodedRows, LogicalSource

# ggml type id -> (name, torch row dtype or None, block elements, block bytes)
_TYPES = {
    0: ("f32", torch.float32, 1, 4),
    1: ("f16", torch.float16, 1, 2),
    30: ("bf16", None, 1, 2),
    142: ("pq2_0", None, 128, 34),
}
HALF_TYPE = 30
PQ2_0_TYPE = 142
PQ2_0_GROUP = 128
PQ2_0_BLOCK_BYTES = 34
PQ2_0_ZERO_CODE = 1

_SCALAR = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I",
    5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}

HalfKind = Literal["bf16", "f16"]


@dataclass(frozen=True, slots=True)
class GgufTensor:
    name: str
    dims: tuple[int, ...]  # ggml order: dims[0] is contiguous, as in ggml ne[]
    ggml_type: int
    offset: int

    @property
    def type_name(self) -> str:
        return _TYPES[self.ggml_type][0]

    @property
    def shape(self) -> tuple[int, ...]:
        """Row-major shape: ggml dims reversed, matching NInfer's [out, in]."""
        return tuple(reversed(self.dims))

    @property
    def elements(self) -> int:
        return prod(self.dims)

    @property
    def blocks(self) -> int:
        block = _TYPES[self.ggml_type][2]
        if self.elements % block:
            raise ValueError(
                f"{self.name}: {self.elements} elements are not a multiple of {block}"
            )
        return self.elements // block

    @property
    def bytes(self) -> int:
        return self.blocks * _TYPES[self.ggml_type][3]


class GgufSource:
    """Read a GGUF checkpoint.

    ``path`` may be a file or a directory containing exactly one ``*.gguf``. The
    interface mirrors `SafetensorsSource` where the conversion pipeline depends on
    it (`has`, `describe`, `read_flat`), and adds exact code import through
    :meth:`read_encoded`.
    """

    def __init__(self, path: str | Path, *, half: HalfKind = "bf16") -> None:
        self.path = Path(path)
        if self.path.is_dir():
            candidates = sorted(self.path.glob("*.gguf"))
            if len(candidates) != 1:
                raise ValueError(
                    f"{self.path}: expected exactly one .gguf, found {len(candidates)}"
                )
            self.path = candidates[0]
        if half not in ("bf16", "f16"):
            raise ValueError("half must be 'bf16' or 'f16'")
        self.half: HalfKind = half
        self.config: dict = {}
        self.bytes_read = 0
        self._stream = self.path.open("rb")
        try:
            self._read_header()
        except Exception:
            self._stream.close()
            raise
        self.weight_map = dict.fromkeys(self.tensors)

    # -- header -----------------------------------------------------------

    def _read_header(self) -> None:
        stream = self._stream

        def u32() -> int:
            return struct.unpack("<I", stream.read(4))[0]

        def u64() -> int:
            return struct.unpack("<Q", stream.read(8))[0]

        def string() -> str:
            return stream.read(u64()).decode("utf-8", errors="replace")

        def value(kind: int):
            if kind == 8:
                return string()
            if kind == 9:
                element = u32()
                return [value(element) for _ in range(u64())]
            try:
                code = _SCALAR[kind]
            except KeyError:
                raise ValueError(f"{self.path}: unknown GGUF value type {kind}") from None
            return struct.unpack(code, stream.read(struct.calcsize(code)))[0]

        if stream.read(4) != b"GGUF":
            raise ValueError(f"{self.path}: not a GGUF file")
        self.version = u32()
        if self.version not in (2, 3):
            raise ValueError(f"{self.path}: unsupported GGUF version {self.version}")
        count = u64()
        kv_count = u64()
        self.metadata: dict[str, object] = {}
        for _ in range(kv_count):
            key = string()
            self.metadata[key] = value(u32())
        tensors: dict[str, GgufTensor] = {}
        for _ in range(count):
            name = string()
            ndim = u32()
            dims = tuple(u64() for _ in range(ndim))
            ggml_type = u32()
            offset = u64()
            if ggml_type not in _TYPES:
                raise ValueError(
                    f"{self.path}: tensor {name!r} uses unsupported ggml type {ggml_type}"
                )
            tensors[name] = GgufTensor(name, dims, ggml_type, offset)
        alignment = int(self.metadata.get("general.alignment", 32))
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError(f"{self.path}: invalid alignment {alignment}")
        self.alignment = alignment
        self.data_offset = (stream.tell() + alignment - 1) // alignment * alignment
        self.tensors = tensors

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "GgufSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- lookup -----------------------------------------------------------

    def has(self, name: str) -> bool:
        return name in self.tensors

    def describe(self, name: str) -> GgufTensor:
        try:
            return self.tensors[name]
        except KeyError:
            raise ValueError(f"{self.path}: missing source tensor {name!r}") from None

    def names(self) -> Iterator[str]:
        return iter(self.tensors)

    # -- payload ----------------------------------------------------------

    def _row_bytes(self, info: GgufTensor) -> int:
        """Physical bytes of one row (one dims[0] span) in the file."""
        if len(info.dims) < 2:
            raise ValueError(f"{info.name}: grouped access requires a matrix")
        return (info.dims[0] // _TYPES[info.ggml_type][2]) * _TYPES[info.ggml_type][3]

    def _read_rows(self, info: GgufTensor, first: int, last: int) -> bytes:
        """Read rows ``[first,last)`` verbatim, where a row is one dims[0] span."""
        rows = info.dims[1] if len(info.dims) > 1 else 1
        if not 0 <= first <= last <= rows:
            raise ValueError(f"{info.name}: row range [{first},{last}) exceeds {rows}")
        stride = self._row_bytes(info)
        self._stream.seek(self.data_offset + info.offset + first * stride)
        data = self._stream.read((last - first) * stride)
        if len(data) != (last - first) * stride:
            raise ValueError(f"{info.name}: short read")
        self.bytes_read += len(data)
        return data

    def _row_values(self, info: GgufTensor, first: int, last: int) -> torch.Tensor:
        """Decode rows as conventional row-major ``[last-first, dims[0]]`` values."""
        raw = self._read_rows(info, first, last)
        kind = info.ggml_type
        if kind == 0:
            return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(
                last - first, info.dims[0]
            )
        if kind == 1:
            return torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(
                last - first, info.dims[0]
            )
        if kind == HALF_TYPE:
            words = torch.frombuffer(bytearray(raw), dtype=torch.uint16).to(torch.int32)
            if self.half == "f16":
                values = (words.to(torch.int16)).view(torch.float16)
            else:
                values = (words << 16).view(torch.float32)
            return values.reshape(last - first, info.dims[0])
        if kind == PQ2_0_TYPE:
            groups = info.dims[0] // PQ2_0_GROUP
            blocks = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
                last - first, groups, PQ2_0_BLOCK_BYTES
            )
            scales = (
                blocks[:, :, :2]
                .contiguous()
                .view(torch.float16)
                .reshape(last - first, groups)
            )
            words = blocks[:, :, 2:].to(torch.int32)
            # code c lives in bits 2*(c%4) of byte c//4, low slot first
            codes = torch.empty(
                (last - first, groups, PQ2_0_GROUP), dtype=torch.int8
            )
            for slot in range(4):
                codes[:, :, slot::4] = (
                    (words >> (2 * slot)) & 0x03
                ).to(torch.int8)
            values = (codes.to(torch.float32) - PQ2_0_ZERO_CODE) * scales.unsqueeze(-1)
            return values.reshape(last - first, info.dims[0])
        raise ValueError(f"{info.name}: no row reader for {info.type_name}")

    def read_flat(self, name: str, begin: int = 0, end: int | None = None) -> torch.Tensor:
        """Read logical elements ``[begin,end)`` in conventional row-major order."""
        info = self.describe(name)
        end = info.elements if end is None else end
        if not 0 <= begin <= end <= info.elements:
            raise ValueError(
                f"{name}: element range [{begin},{end}) exceeds {info.elements}"
            )
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        if len(info.dims) < 2:
            # A vector has no group structure; read its whole payload.
            values = self._vector_values(info)
            return values[begin:end]
        width = info.dims[0]
        first, last = begin // width, (end + width - 1) // width
        block = self._row_values(info, first, last).reshape(-1)
        return block[begin - first * width : end - first * width]

    def _vector_values(self, info: GgufTensor) -> torch.Tensor:
        """Decode a rank-1 tensor (scales, norms and GDN vectors)."""
        self._stream.seek(self.data_offset + info.offset)
        raw = self._stream.read(info.bytes)
        if len(raw) != info.bytes:
            raise ValueError(f"{info.name}: short read")
        self.bytes_read += len(raw)
        kind = info.ggml_type
        if kind == 0:
            return torch.frombuffer(bytearray(raw), dtype=torch.float32)
        if kind == 1:
            return torch.frombuffer(bytearray(raw), dtype=torch.float16).to(torch.float32)
        if kind == HALF_TYPE:
            words = torch.frombuffer(bytearray(raw), dtype=torch.uint16).to(torch.int32)
            if self.half == "f16":
                return words.to(torch.int16).view(torch.float16).to(torch.float32)
            return (words << 16).view(torch.float32)
        raise ValueError(f"{info.name}: no vector reader for {info.type_name}")

    def read_encoded(self, name: str, first: int, last: int) -> EncodedRows:
        """Return stored PQ2_0 codes and fp16 group scales for rows ``[first,last)``.

        Codes keep the checkpoint's unsigned grid index, which is exactly the
        ``q2_g128_fp16`` code, so nothing is requantized.
        """
        info = self.describe(name)
        if info.ggml_type != PQ2_0_TYPE:
            raise ValueError(
                f"{name}: {info.type_name} has no importable ternary codes"
            )
        if len(info.dims) != 2 or info.dims[0] % PQ2_0_GROUP:
            raise ValueError(f"{name}: PQ2_0 rows must be a multiple of {PQ2_0_GROUP}")
        raw = self._read_rows(info, first, last)
        groups = info.dims[0] // PQ2_0_GROUP
        blocks = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
            last - first, groups, PQ2_0_BLOCK_BYTES
        )
        scales = (
            blocks[:, :, :2].contiguous().view(torch.float16).reshape(last - first, groups)
        )
        words = blocks[:, :, 2:].to(torch.int32)
        codes = torch.empty((last - first, groups, PQ2_0_GROUP), dtype=torch.int8)
        for slot in range(4):
            codes[:, :, slot::4] = ((words >> (2 * slot)) & 0x03).to(torch.int8)
        return EncodedRows(
            "q2_g128_fp16",
            codes,
            scales,
        )


def tensor_source(
    store: GgufSource,
    name: str,
    shape: tuple[int, ...],
    *,
    offset: int = 0,
    source_shape: tuple[int, ...] | None = None,
) -> LogicalSource:
    """A value view over one GGUF tensor, mirroring the Safetensors helper."""
    expected = shape if source_shape is None else source_shape

    def read(begin: int, end: int) -> torch.Tensor:
        info = store.describe(name)
        if info.shape != expected:
            raise ValueError(
                f"{name}: expected source shape {expected}, got {info.shape}"
            )
        return store.read_flat(name, offset + begin, offset + end)

    def encoded(begin: int, end: int) -> EncodedRows:
        info = store.describe(name)
        if len(shape) != 2 or info.shape[1] != shape[1] or offset % shape[1]:
            raise ValueError(f"{name}: encoded view must cover complete source rows")
        first = offset // shape[1] + begin
        return store.read_encoded(name, first, first + end - begin)

    info = store.describe(name)
    encoded_reader = encoded if info.ggml_type == PQ2_0_TYPE else None
    return LogicalSource(
        shape, f"{store.path}:{name}[{offset}:{offset+prod(shape)}]", read,
        encoded_reader,
    )
