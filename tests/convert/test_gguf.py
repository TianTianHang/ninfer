from __future__ import annotations

import struct

import torch

from tools.convert.bonsai_mapping import layer_mappings
from tools.convert.sources.gguf import GgufSource, tensor_source
from tools.convert.sources.logical import array_source, transpose_source


def _string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _gguf(path, tensors, payloads, metadata=None):
    metadata = {"general.alignment": 32, **(metadata or {})}
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, value in metadata.items():
        header += _string(key)
        if isinstance(value, int):
            header += struct.pack("<II", 4, value)
        else:
            raise AssertionError("test metadata only needs uint32 values")
    for name, dims, kind in tensors:
        header += _string(name)
        header += struct.pack("<I", len(dims))
        header += b"".join(struct.pack("<Q", dim) for dim in dims)
        header += struct.pack("<IQ", kind, 0)
    header += b"\0" * ((32 - len(header) % 32) % 32)
    path.write_bytes(header + b"".join(payloads))


def test_pq2_0_codes_and_scales_are_imported_verbatim(tmp_path):
    block0 = struct.pack("<e", 1.0) + bytes([0xE4]) + bytes(31)
    block1 = struct.pack("<e", 2.0) + bytes([0x1B]) + bytes(31)
    path = tmp_path / "pq2.gguf"
    _gguf(path, [("proj.weight", (128, 2), 142)], [block0 + block1])
    with GgufSource(path) as source:
        encoded = source.read_encoded("proj.weight", 0, 2)
        assert encoded.format == "q2_g128_fp16"
        assert encoded.codes.shape == (2, 1, 128)
        assert encoded.codes[0, 0, :4].tolist() == [0, 1, 2, 3]
        assert encoded.codes[1, 0, :4].tolist() == [3, 2, 1, 0]
        assert encoded.scales.tolist() == [[1.0], [2.0]]
        logical = tensor_source(source, "proj.weight", (2, 128))
        assert torch.equal(logical.read_encoded(0, 2).codes, encoded.codes)
        assert logical.values(0, 4).tolist() == [-1.0, 0.0, 1.0, 2.0]


def test_declared_bf16_payload_can_be_read_as_fp16_bits(tmp_path):
    values = struct.pack("<HH", 0x3E00, 0xB800)  # FP16 1.5, -0.5
    path = tmp_path / "half.gguf"
    _gguf(path, [("blk.0.ssm_alpha.weight", (2,), 30)], [values])
    with GgufSource(path, half="f16") as source:
        assert torch.equal(
            source.read_flat("blk.0.ssm_alpha.weight"),
            torch.tensor([1.5, -0.5]),
        )
    with GgufSource(path, half="bf16") as source:
        assert source.read_flat("blk.0.ssm_alpha.weight")[0].item() != 1.5


def test_bonsai_mapping_preserves_interleaved_attention_and_gdn_spans():
    full = layer_mappings(3)
    query = full["text/layers/3/attention/query"].rows
    gate = full["text/layers/3/attention/gate"].rows
    assert len(query) == 24 and query[0].begin == 0 and query[0].end == 256
    assert gate[0].begin == 256 and gate[0].end == 512
    gdn = layer_mappings(0)
    assert (gdn["text/layers/0/gdn/query"].rows[0].begin,
            gdn["text/layers/0/gdn/query"].rows[0].end) == (0, 2048)
    assert (gdn["text/layers/0/gdn/value"].rows[0].begin,
            gdn["text/layers/0/gdn/value"].rows[0].end) == (4096, 10240)


def test_transpose_source_transposes_matrix_axes_not_flat_storage():
    source = array_source(torch.arange(6, dtype=torch.float32).reshape(3, 2), "conv")
    transposed = transpose_source(source, (1, 0), (2, 3))
    assert transposed.values().reshape(2, 3).tolist() == [[0, 2, 4], [1, 3, 5]]
