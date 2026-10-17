"""Production converter adapter for the Ternary-Bonsai PQ2_0/PTQ1_0 GGUF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch

from .bonsai_mapping import (
    FFN_WIDTH,
    HEAD_DIM,
    HIDDEN,
    KEY_DIM,
    LAYERS,
    N_KV_HEADS,
    N_Q_HEADS,
    N_V_HEADS,
    VALUE_DIM,
    all_mappings,
    expected_shape,
    is_full_attention,
)
from .model import Model, Parameter
from .sources.gguf import GgufSource, tensor_source
from .sources.logical import LogicalSource, array_source, select_rows, transpose_source



def _text_config() -> dict:
    return {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text",
        "hidden_size": HIDDEN,
        "vocab_size": 248320,
        "num_hidden_layers": LAYERS,
        "max_position_embeddings": 262144,
        "tie_word_embeddings": False,
        "rms_norm_eps": 1e-6,
        "layer_types": [
            "full_attention" if is_full_attention(i) else "linear_attention"
            for i in range(LAYERS)
        ],
        "num_attention_heads": N_Q_HEADS,
        "num_key_value_heads": N_KV_HEADS,
        "head_dim": HEAD_DIM,
        "rope_parameters": {
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
        },
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_num_value_heads": N_V_HEADS,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "intermediate_size": FFN_WIDTH,
        "hadamard": {
            "block_size": 1024,
            "sign_widths": [HIDDEN, 6144, FFN_WIDTH],
            "gdn_v_grouped": True,
        },
    }


def _resources(source: GgufSource, overrides: Mapping[str, str | Path] | None):
    meta = source.metadata
    tokens = list(meta["tokenizer.ggml.tokens"])
    types = list(meta.get("tokenizer.ggml.token_type", [1] * len(tokens)))
    merges = list(meta.get("tokenizer.ggml.merges", []))
    vocab = {}
    added = []
    for index, (token, kind) in enumerate(zip(tokens, types)):
        if kind in (1, 6):
            vocab[token] = index
        else:
            added.append({
                "id": index,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": kind in (2, 3),
            })
    merge_pairs = []
    for item in merges:
        left, separator, right = item.partition(" ")
        if not separator:
            raise ValueError(f"malformed GGUF merge {item!r}")
        merge_pairs.append([left, right])
    added_decoder = {str(item["id"]): item for item in added}
    pad_id = int(meta["tokenizer.ggml.padding_token_id"])
    if tokens[pad_id] != "<|endoftext|>":
        raise ValueError(f"unexpected padding token {tokens[pad_id]!r}")
    special = {
        "eos_token": tokens[int(meta["tokenizer.ggml.eos_token_id"])],
        "bos_token": tokens[int(meta["tokenizer.ggml.bos_token_id"])],
        "pad_token": tokens[pad_id],
    }
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merge_pairs,
        },
    }
    tokenizer_config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "clean_up_tokenization_spaces": False,
        "model_max_length": 262144,
        "added_tokens_decoder": added_decoder,
        **special,
    }
    generation = {
        "temperature": float(meta.get("general.sampling.temp", 1.0)),
        "top_p": float(meta.get("general.sampling.top_p", 0.95)),
        "top_k": int(meta.get("general.sampling.top_k", 20)),
        "do_sample": True,
        "repetition_penalty": 1.0,
        "bos_token_id": int(meta["tokenizer.ggml.bos_token_id"]),
        "eos_token_id": int(meta["tokenizer.ggml.eos_token_id"]),
        "pad_token_id": pad_id,
    }
    template = meta.get("tokenizer.chat_template", "")
    if not template:
        raise ValueError("the GGUF checkpoint carries no chat template")

    def dump(value) -> bytes:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()

    values = {
        "tokenizer.json": dump(tokenizer),
        "tokenizer_config.json": dump(tokenizer_config),
        "chat_template.jinja": str(template).encode(),
        "generation_config.json": dump(generation),
    }
    overrides = {} if overrides is None else dict(overrides)
    unknown = set(overrides) - set(values)
    if unknown:
        raise ValueError(f"resource overrides have no selected consumer: {sorted(unknown)}")
    for role, path in overrides.items():
        values[role] = Path(path).read_bytes()
    refs = {"text": {role: f"resource/text/{role}" for role in values}}
    payloads = {f"resource/text/{role}": data for role, data in values.items()}
    # Keep the public-domain validation in the same place as safetensors builds.
    from .resources import token_domain

    count, special_ids = token_domain(tokenizer, tokenizer_config, len(tokens))
    return refs, payloads, count, special_ids


def _mapped_source(source: GgufSource, destination: str, mapping) -> LogicalSource:
    shape = expected_shape(destination)
    if shape is None:
        raise ValueError(f"{destination}: no expected logical shape")
    info = source.describe(mapping.source)
    view = tensor_source(source, mapping.source, info.shape)
    if mapping.rows is not None:
        view = select_rows(view, tuple((span.begin, span.end) for span in mapping.rows))
    if mapping.transpose:
        view = transpose_source(view, (1, 0), shape)
    if mapping.transform == "ggml_ssm_a_to_a_log":
        base_view = view
        def read(begin: int, end: int) -> torch.Tensor:
            values = base_view.values(begin, end).to(torch.float32)
            if bool((values > 0).any()):
                raise ValueError(f"{destination}: ssm_a must be non-positive")
            return torch.log(-values)
        view = LogicalSource(shape, f"log(-{view.label})", read)
    elif mapping.transform is not None:
        raise ValueError(f"{destination}: unknown transform {mapping.transform!r}")
    if view.shape != shape:
        raise ValueError(f"{destination}: mapped shape {view.shape} != {shape}")
    return view


def _inputs(destination: str) -> tuple[str, ...]:
    if destination in ("text/token_embedding",) or destination.endswith(
        ("/input_norm", "/post_attention_norm", "/attention/query_norm", "/attention/key_norm",
         "/gdn/a_log", "/gdn/dt_bias", "/gdn/convolution", "/gdn/norm", "/final_norm")
    ):
        return ()
    if destination == "text/output_head":
        return ("text/final_hidden",)
    layer = destination.split("/layers/", 1)[1].split("/", 1)[0]
    prefix = f"text/layers/{layer}/"
    if "/mlp/" in destination:
        return (prefix + ("mlp/product" if destination.endswith("/down") else "ffn_input"),)
    if "/attention/" in destination:
        return (prefix + ("attention/gated_output" if destination.endswith("/output") else "mixer_input"),)
    if "/gdn/" in destination:
        return (prefix + ("gdn/gated_output" if destination.endswith("/output") else "mixer_input"),)
    raise ValueError(f"{destination}: cannot infer model input")


def _packing_groups() -> list[tuple[str, ...]]:
    groups = []
    for layer in range(LAYERS):
        p = f"text/layers/{layer}/"
        groups.append((p + "mlp/gate", p + "mlp/up"))
        if is_full_attention(layer):
            groups.append(tuple(p + "attention/" + role for role in ("query", "key", "gate", "value")))
        else:
            groups.append((p + "gdn/query", p + "gdn/key"))
            groups.append((p + "gdn/value", p + "gdn/z"))
        groups.append((p + "gdn/a_projection", p + "gdn/b_projection")) if not is_full_attention(layer) else None
    return groups


def build_model(source: GgufSource, *, resource_overrides=None) -> Model:
    if source.describe("token_embd.weight").type_name not in ("pq2_0", "ptq1_0"):
        raise ValueError("this adapter requires the Bonsai PQ2_0 or PTQ1_0 token embedding")
    refs, resources, token_count, special_ids = _resources(source, resource_overrides)
    config = _text_config()
    model = Model({"text": {"config": config, "resources": refs["text"]}}, resources=resources,
                  packing_groups=_packing_groups(), token_count=token_count,
                  special_token_ids=special_ids)
    for destination, mapping in sorted(all_mappings().items()):
        shape = expected_shape(destination)
        view = _mapped_source(source, destination, mapping)
        direct = "fp32" if destination.endswith(("/gdn/a_log", "/gdn/dt_bias")) else "bf16"
        model.add(Parameter(destination, shape, view, inputs=_inputs(destination), direct_format=direct))
    widths = source.metadata.get("prism.hadamard.sign_widths")
    values = source.metadata.get("prism.hadamard.sign_values")
    if widths != [HIDDEN, 6144, FFN_WIDTH] or not isinstance(values, list):
        raise ValueError("GGUF Hadamard metadata does not match the Bonsai geometry")
    cursor = 0
    for width in widths:
        signs = values[cursor : cursor + width]
        if len(signs) != width or any(value not in (-1, 1) for value in signs):
            raise ValueError("GGUF Hadamard sign values must be complete +/-1 vectors")
        name = f"text/hadamard_signs/{width}"
        model.add(Parameter(name, (width,), array_source(torch.tensor(signs, dtype=torch.int32), name),
                            direct_format="int32"))
        cursor += width
    if cursor != len(values):
        raise ValueError("GGUF Hadamard sign widths do not cover sign values")
    return model


def recipe(model, recipe, sources=None):
    from .methods import import_encoded
    for name, parameter in model.parameters.items():
        if parameter.source.read_encoded is not None:
            encoded_format = parameter.source.read_encoded(0, 1).format
            recipe.assign(name, format=encoded_format, method=import_encoded)


__all__ = ["build_model", "recipe"]
