"""Map a Ternary-Bonsai GGUF onto NInfer's Qwen3.5 Dense parameter names.

Geometry facts taken from the checkpoint metadata and PrismML's `qwen35.cpp`:

    hidden 5120      layers 64        full-attention interval 4
    head_k_dim = state_size = 128    n_k_heads = group_count = 16
    head_v_dim = state_size = 128    n_v_heads = time_step_rank = 48
    key_dim = 2048   value_dim = 6144   conv_dim = key_dim*2 + value_dim = 10240

The GGUF stores a folded basis for 401 projection tensors and one inverse-folded
embedding table; that fold is part of the model function and is applied to the
activations at run time, so this map copies weights unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

# One mapped output: a destination parameter plus the exact transform to apply.
@dataclass(frozen=True, slots=True)
class RowSpan:
    """Take rows [begin,end) of the source tensor."""

    begin: int
    end: int


@dataclass(frozen=True, slots=True)
class Mapping:
    """How one destination parameter is produced from the checkpoint."""

    source: str
    # Rows of the source, in destination order. ``None`` means the whole tensor.
    rows: tuple[RowSpan, ...] | None = None
    # A rank-2 tensor stored the other way round (the GDN convolution kernel).
    transpose: bool = False
    # A checkpoint tensor reached through a pointwise function.
    transform: str | None = None


HIDDEN = 5120
LAYERS = 64
FULL_ATTENTION_INTERVAL = 4
HEAD_DIM = 256
N_Q_HEADS = 24
N_KV_HEADS = 4
N_V_HEADS = 48
KEY_DIM = 2048
VALUE_DIM = 6144
Q_DIM = N_Q_HEADS * HEAD_DIM  # 6144
KV_DIM = N_KV_HEADS * HEAD_DIM  # 1024


def is_full_attention(layer: int) -> bool:
    """Full attention every 4th block; the rest are GDN, matching layer_types."""
    return (layer + 1) % FULL_ATTENTION_INTERVAL == 0


def _span(begin: int, end: int) -> tuple[RowSpan, ...]:
    return (RowSpan(begin, end),)


def _layer(layer: int, base: str = "text") -> str:
    return f"{base}/layers/{layer}/"


def global_mappings() -> dict[str, Mapping]:
    return {
        "text/token_embedding": Mapping("token_embd.weight"),
        "text/output_head": Mapping("output.weight"),
        "text/final_norm": Mapping("output_norm.weight"),
    }


def layer_mappings(layer: int) -> dict[str, Mapping]:
    """Destination parameter -> source mapping for one Text block."""
    p = _layer(layer)
    blk = f"blk.{layer}."
    out: dict[str, Mapping] = {
        # Block norms are plain RMS norms stored as F32.
        p + "input_norm": Mapping(blk + "attn_norm.weight"),
        p + "post_attention_norm": Mapping(blk + "post_attention_norm.weight"),
        # The Dense feed-forward block.
        p + "mlp/gate": Mapping(blk + "ffn_gate.weight"),
        p + "mlp/up": Mapping(blk + "ffn_up.weight"),
        p + "mlp/down": Mapping(blk + "ffn_down.weight"),
    }
    if is_full_attention(layer):
        # attn_q packs query and gate interleaved per head: head h occupies rows
        # [h*2*head_dim, (h+1)*2*head_dim), query first, then that head's gate.
        # Splitting the matrix in half instead would change the model.
        query_rows = tuple(
            RowSpan(h * 2 * HEAD_DIM, h * 2 * HEAD_DIM + HEAD_DIM) for h in range(N_Q_HEADS)
        )
        gate_rows = tuple(
            RowSpan(h * 2 * HEAD_DIM + HEAD_DIM, (h + 1) * 2 * HEAD_DIM)
            for h in range(N_Q_HEADS)
        )
        out[p + "attention/query"] = Mapping(blk + "attn_q.weight", rows=query_rows)
        out[p + "attention/gate"] = Mapping(blk + "attn_q.weight", rows=gate_rows)
        out[p + "attention/key"] = Mapping(blk + "attn_k.weight")
        out[p + "attention/value"] = Mapping(blk + "attn_v.weight")
        out[p + "attention/output"] = Mapping(blk + "attn_output.weight")
        out[p + "attention/query_norm"] = Mapping(blk + "attn_q_norm.weight")
        out[p + "attention/key_norm"] = Mapping(blk + "attn_k_norm.weight")
        return out

    # A GDN block packs its input projection as [query(2048) | key(2048) | value(6144)].
    # ssm_conv1d is [4, conv_dim] in ggml order and NInfer's logical convolution is
    # [taps, channels] over the same data, so only the axes are swapped.
    out[p + "gdn/convolution"] = Mapping(blk + "ssm_conv1d.weight", transpose=True)
    out[p + "gdn/a_projection"] = Mapping(blk + "ssm_alpha.weight")
    out[p + "gdn/b_projection"] = Mapping(blk + "ssm_beta.weight")
    # ggml stores ssm_a = -exp(A_log); NInfer's formula is -exp(A_log[h]), so the
    # stored value has to be converted. Range check below.
    out[p + "gdn/a_log"] = Mapping(blk + "ssm_a", transform="ggml_ssm_a_to_a_log")
    out[p + "gdn/dt_bias"] = Mapping(blk + "ssm_dt.bias")
    out[p + "gdn/norm"] = Mapping(blk + "ssm_norm.weight")
    # in_proj_qkv packs [query(2048) | key(2048) | value(6144)].
    out[p + "gdn/query"] = Mapping(blk + "attn_qkv.weight", rows=_span(0, KEY_DIM))
    out[p + "gdn/key"] = Mapping(
        blk + "attn_qkv.weight", rows=_span(KEY_DIM, 2 * KEY_DIM)
    )
    out[p + "gdn/value"] = Mapping(
        blk + "attn_qkv.weight", rows=_span(2 * KEY_DIM, 2 * KEY_DIM + VALUE_DIM)
    )
    out[p + "gdn/z"] = Mapping(blk + "attn_gate.weight")
    out[p + "gdn/output"] = Mapping(blk + "ssm_out.weight")
    return out


def all_mappings() -> dict[str, Mapping]:
    result = global_mappings()
    for layer in range(LAYERS):
        result.update(layer_mappings(layer))
    return result


FFN_WIDTH = 17408
VOCAB = 248320


def expected_shape(destination: str) -> tuple[int, ...] | None:
    """The logical [out, in] shape NInfer binds for one parameter."""
    if destination in ("text/token_embedding", "text/output_head"):
        return (VOCAB, HIDDEN)
    if destination == "text/final_norm":
        return (HIDDEN,)
    if "/mlp/" in destination:
        role = destination.rsplit("/", 1)[1]
        return (HIDDEN, FFN_WIDTH) if role == "down" else (FFN_WIDTH, HIDDEN)
    if "/attention/" in destination:
        role = destination.rsplit("/", 1)[1]
        if role in ("query", "gate"):
            return (Q_DIM, HIDDEN)
        if role in ("key", "value"):
            return (KV_DIM, HIDDEN)
        if role in ("query_norm", "key_norm"):
            return (HEAD_DIM,)
        if role == "output":
            return (HIDDEN, Q_DIM)
    if "/gdn/" in destination:
        role = destination.rsplit("/", 1)[1]
        if role in ("query", "key"):
            return (KEY_DIM, HIDDEN)
        if role in ("value", "z"):
            return (VALUE_DIM, HIDDEN)
        if role in ("a_projection", "b_projection"):
            return (N_V_HEADS, HIDDEN)
        if role in ("a_log", "dt_bias"):
            return (N_V_HEADS,)
        if role == "norm":
            return (128,)
        if role == "convolution":
            return (4, 10240)
        if role == "output":
            return (HIDDEN, VALUE_DIM)
    if destination.endswith(("/input_norm", "/post_attention_norm")):
        return (HIDDEN,)
    return None
