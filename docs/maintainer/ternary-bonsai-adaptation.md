# Ternary Bonsai adaptation notes

This document records the maintenance contract and verification lessons for adapting
PrismML's `Ternary-Bonsai-2-27B` checkpoint to NInfer. It is a debugging and
qualification reference, not a user-facing model card.

## Checkpoint geometry

The dense text model uses:

- hidden size 5120, 64 layers, vocabulary 248320, FFN width 17408;
- 16 full-attention layers and 48 GDN layers;
- full attention with 24 query heads, 4 KV heads, and head dimension 256;
- GDN with 16 key heads of dimension 128 and 48 value heads of dimension 128;
- convolution width 4 and 10240 convolution channels.

The attention and GDN layer pattern must be taken from the checkpoint contract: full
attention occurs at layers 3, 7, 11, and so on. A plausible output from a wrong layer
pattern is not evidence of a correct conversion.

## PQ2_0 and q2_g128_fp16

GGUF type 142 is PrismML's custom `PQ2_0`; it is not GGML `TQ2_0`.

Each PQ2_0 block contains 128 values:

1. one FP16 scale;
2. 32 code bytes;
3. four low-slot-first 2-bit codes per byte.

The reconstruction contract is:

```text
value = (code - 1) * fp16_scale
```

Therefore the codebook is `{0, 1, 2, 3} -> {-1, 0, +1, +2}`. The NInfer
`q2_g128_fp16` RowSplit representation preserves the grouped codes and FP16 scales.
Its conversion path must use `import_encoded`; ordinary absmax quantization is not a
lossless import path for this format.

The source reader exposes encoded PQ2 rows as `[rows, groups, 128]`. Flattening this
shape too early makes a converter test pass at the wrong abstraction level and can
hide group or plane-order errors.

Packing groups must match the runtime parent-row contract:

- MLP: gate rows followed by up rows;
- full attention: query, key, gate, value;
- GDN: query, key and value, z as separate paired parents;
- GDN control projections: a projection followed by b projection.

Full-attention query and gate rows are interleaved in the source projection. They must
be selected into their logical destination ranges before packing. This is a logical
mapping operation, not a physical artifact reorder.

## PTQ1_0 and native ptq1_g128_fp16

GGUF type 143 is PrismML's `PTQ1_0`, not the upstream `TQ1_0` block size. Each
128-value block is 28 bytes:

1. `qs[24]`: five base-3 digits per byte for values 0 through 119;
2. `qh[2]`: four base-3 digits per byte for values 120 through 127;
3. `d`: one FP16 scale at byte offset 26.

For a stored byte `b`, digit `n` is decoded by repeatedly applying
`b = (b * 3) & 0xff` and extracting `(b * 3) >> 8 - 1`. The decoded coefficient
is in `{-1, 0, +1}` and the value is `coefficient * d`. This is the exact decoder
used by the local llama.cpp Prism branch. It is not an ordinary absmax
requantization and must not be converted through a floating-point approximation.

NInfer stores PTQ1 as `ptq1_g128_fp16` in the same RowSplit object contract, but
keeps the 28-byte blocks intact. The C++ linear, embedding, Attention input,
GDN input, residual linear-add, and SwiGLU routes decode the blocks directly.
Hadamard activation preparation is shared with the existing ternary path; only
the weight decoder differs. The converter's `import_encoded` method preserves
FP16 scales and re-encodes the base-3 digits using the llama.cpp ceiling rule.

The independent oracle for PTQ1 must check all 128 lane positions, including the
16-byte, 8-byte, and 2-byte staging regions, exact FP16 scale bits, and direct
operator results against the llama.cpp reference decoder. A successful artifact
load or plausible generation result is not sufficient.


Prism folding uses normalized, self-inverse Sylvester-Walsh Hadamard transforms with
runtime block size 1024 and explicit sign vectors of widths 5120, 6144, and 17408.
The operation order is part of the model contract:

```text
forward folded projection:  H(sign * x)
inverse token embedding:   sign * H(z)
```

The output projection uses the forward transform. GDN value output needs the runtime
regrouping `[head_dim, key_heads, replication] -> [head_dim, replication, key_heads]`
before the folded projection. GDN alpha and beta control projections are ordinary
projections and must not receive the folded activation transform.

## GDN and normalization semantics

GGUF type 30 in this checkpoint is BF16. In particular, GDN alpha and beta projection
weights must be read as BF16 rather than FP16. The GGUF `ssm_a` values are negative
and are stored in NInfer as:

```text
A_log = log(-ssm_a)
```

The control equations are:

```text
g    = -exp(A_log) * softplus(alpha + dt_bias)
beta = sigmoid(beta_projection)
```

The block RMSNorm weights in this GGUF are ordinary multipliers. They are not delta
weights. The correct operation is `normalized * weight`, not
`normalized * (1 + weight)`. This must hold in the regular RMSNorm calls and in the
fused GDN SIMT and cooperative MMA paths. A hidden `unit_offset` is a semantic bug,
not a harmless numerical approximation.

The independent Prism graph was essential here: before the fix, the first-layer
`attn_norm` sum and GDN qkv sum diverged substantially even though artifact loading,
CUDA dispatch, and generation all completed successfully.

## Qualification protocol

Every new representation or folded execution route should be qualified in this order:

1. Run the independent RowSplit oracle against the source quantizer. Check code-byte
   packing, decode values, FP16 scales, and the ternary grid.
2. Verify every logical mapping shape and layer composition.
3. Compare raw PQ2 code and scale planes for representative parameters from token
   embedding, output head, MLP, attention, GDN query/key/value/z, and output weights.
4. Compare every direct artifact tensor after applying only documented transforms such
   as `A_log`; do not compare only summary statistics.
5. Use the reference implementation with the exact same tokenizer output, chat
   template, reasoning setting, sampling mode, and context settings.
6. Add a reference graph callback or equivalent intermediate dump for the first layer.
   Compare an input-normalized node and the first folded projection output before
   investigating later recurrent state or sampling behavior.
7. Compare generated token IDs and text. Successful startup, finite tensors, or a
   plausible sentence are insufficient evidence.

For this adaptation, the completed checks were:

- 963 mapped parameter shapes, including 514 PQ2_0 parameters;
- all 16 full-attention and 48 GDN layer mappings;
- 449 direct mappings compared exactly after documented transforms;
- independent Q2 RowSplit oracle passed all checks;
- Python artifact and converter tests passed;
- the Prism reference and NInfer produced the same greedy response for the same
  13-token no-thinking prompt.

## Build reproducibility

With the Nix CUDA 13.1 toolchain and Ninja, compiler-generated dependency tracking
left stale `.ninja_deps` records and caused every CUDA object to rebuild. The project
sets `CMAKE_DEPENDS_USE_COMPILER=OFF` for Ninja so CMake owns dependency scanning.
After the first migration build, an unchanged build must report:

```text
ninja: no work to do.
```

Use an explicit build directory and target when verifying a change:

```bash
cmake --build build --parallel 4 --target ninfer
```
