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

The native contraction has two private routes. For a token extent `T <= 8`, a
warp owns one code byte per lane for a small row group (four rows for
`N >= 6144`, otherwise one), decodes each byte once, and reuses the decoded
digits across those rows and all `T` columns. For `T > 8`, a 96-row by
64-column CTA tile (768 threads, 24 warps, six 16-row bands by four 16-column
bands; shared `a_tile[96][136]`, `b_tile[64][136]`, two CTAs per SM) decodes one
weight group into shared memory and reuses it across the staged columns. Each
warp computes a 16x16 output tile as two `m16n8k16` `mma.sync` operations, so
the decoded bf16 weights and bf16 activation tile are contracted on the tensor
cores under FP32 accumulation, and the group scale is applied to the completed
group product exactly as in the warp route. The tensor-core route is only
valid for `sm_80+`; this build is `sm_120a` only. Both routes multiply the
completed group dot product by the group scale, so the scale is applied once per
group rather than once per weight. The fused SwiGLU route stages the gate/up
projection through the same contraction and applies the SiLU product in one
reduce pass; the earlier dedicated fused PTQ1 kernel no longer exists.

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

## Local performance on the RTX 5060 Laptop

### Scope and conditions

These numbers are for the local qualification machine and are not the fork's RTX 5090 reference:
an 8 GB RTX 5060 Laptop GPU (26 SMs, `sm_120a`), NVIDIA driver 595.104.02 with the CUDA 13.1
toolkit, and the 27B `out/Ternary-Bonsai-2-27B-PTQ1_0.ninfer` PTQ1 artifact. Tuned choices (row
groups for `T <= 8`, the 96x64x128 tensor-core tile, and the power state below) are specific to
this GPU; do not extrapolate them to a 5090 or to other mobile parts without re-measuring.

End-to-end runs use `ninfer_bench` with `--max-ctx 2048`, BF16 KV, CUDA Graph execution, one
execution lane, greedy decoding, and the `bench_corpus.ids` prompt set. Op runs use
`ninfer_linear_bench` with a cold-cache flush per iteration and report the median.

```bash
LD_LIBRARY_PATH=/run/opengl-driver/lib ./build/bench/ninfer_bench \
  --weights out/Ternary-Bonsai-2-27B-PTQ1_0.ninfer -p 512,1024,2048 -n 128 -r 3 --warmup 1 --max-ctx 2048
LD_LIBRARY_PATH=/run/opengl-driver/lib ./build/bench/ninfer_linear_bench \
  --qtype ptq1 --n 34816 --k 5120 --t 512 --repeat 5
```

### Power state

The GPU reports a default power limit of 50 W and a maximum of 115 W. At the default limit the SM
clock settles near 1.5 GHz under a memory-heavy decode load. Dynamic Boost raises the operating
limit (about 100 W at idle, up to the 115 W ceiling under load) and the SM clock to roughly
2.5-2.75 GHz. On Linux the mechanism is the `nvidia-powerd` daemon; without it the driver stays at
the default limit even though `nvidia-smi` reports the higher maximum. On NixOS enable it with
`hardware.nvidia.dynamicBoost.enable = true;`. `nvidia-smi -pl`, the boost slider, and clock
locking do not work on this mobile vBIOS. Record which limit was active when comparing runs.

Measured power effect on this model, holding the routes fixed:

| State | Decode (t/s) | pp512 (t/s) |
|---|---:|---:|
| 50 W default, optimized routes | 11.73 | 71.8 |
| 100 W Dynamic Boost, same routes | 21.0 | 98.7 |

### Routes and dispatch

- `T <= 8`: warp contraction. Lane `j` owns code byte `j` of the group; four output rows per warp
  when `N >= 6144`, otherwise one. A dedicated `T = 1` kernel is used when the aligned variant applies and `N % (rows * 8) == 0`.
- `T > 8`: tensor-core block contraction (96 rows x 64 tokens x 128 lanes per group, two CTAs per
  SM). Ternary codes are decoded to exact bf16 `{-1,0,+1}` (Q2 additionally `+2`) in shared memory;
  the activation tile is bf16; contraction uses BF16 `mma.sync` with FP32 accumulation. This keeps
  the A16 arithmetic contract unchanged.
- The per-128-lane group scale is applied to the completed group product, so it is applied once per
  group rather than once per weight. The fused SwiGLU route stages gate/up through the same
  contraction and applies SiLU in one reduce pass; the residual linear-add route accumulates
  through the same kernels.

### End-to-end results

`pp` is prefill throughput, `tg` is decode throughput; both in tokens per second.

| Test | Baseline (50 W, scalar routes) | Final (100 W, tensor-core prefill) | Speedup |
|---|---:|---:|---:|
| pp512 | 7.65 | 389.7 +/- 1.0 | 50.9x |
| pp1024 | - | 392.9 +/- 0.7 | - |
| pp2048 | - | 389.6 +/- 0.7 | - |
| tg128 | 6.26 | 20.8 +/- 0.0 | 3.3x |

Progression on this machine: scalar routes at 50 W (6.26 / 7.65), warp routes plus row-group
tuning at 50 W (11.73 / 71.8), Dynamic Boost at 100 W (21.0 / 98.7), then tensor-core prefill
(20.8 / 389.7). Reported peak workspace for this artifact is 132 MiB.

### Op results: decode (`T = 1`, PTQ1)

Median microseconds and effective weight bandwidth over 20 cold-cache repetitions.

| N x K | Microseconds | GB/s |
|---|---:|---:|
| 1024 x 5120 | 16.4 | 70.8 |
| 2048 x 5120 | 26.6 | 86.7 |
| 5120 x 6144 | 67.6 | 102.2 |
| 6144 x 5120 | 63.5 | 108.7 |
| 14336 x 5120 | 122.9 | 131.0 |
| 5120 x 17408 | 180.3 | 108.4 |
| 34816 x 5120 | 266.2 | 146.8 |
| 248320 x 5120 | 1766.4 | 157.7 |

Q2 at `T = 1`: 5120 x 17408 is 139.3 us (170.3 GB/s) and 34816 x 5120 is 249.9 us (189.8 GB/s).
The memory floor at 345 GB/s is about 17 ms per token; decode is instruction-bound rather than
bandwidth-bound, at roughly five decode/MAC instructions per weight.

### Op results: prefill (`T = 512`)

Median milliseconds and logical TFLOP/s over three cold-cache repetitions.

| Format | N x K | Milliseconds | TFLOP/s |
|---|---|---:|---:|
| PTQ1 | 34816 x 5120 | 9.12 | 20.02 |
| PTQ1 | 5120 x 17408 | 4.82 | 18.95 |
| PTQ1 | 14336 x 5120 | 3.45 | 21.77 |
| PTQ1 | 5120 x 6144 | 1.61 | 19.95 |
| Q2 | 34816 x 5120 | 7.77 | 23.49 |
| Q2 | 5120 x 17408 | 4.13 | 22.11 |
| Q2 | 14336 x 5120 | 2.81 | 26.76 |
| Q2 | 5120 x 6144 | 1.36 | 23.65 |

The measured BF16 tensor-core peak on this GPU is about 37.6 TFLOP/s at 2.36 GHz (higher at the
2.55 GHz boost clock), so the prefill route reaches roughly 50-60 % of the tensor-core peak. The
remaining prefill cost is dominated by the ternary decode in the staging phase; the token tile was
kept at 64 because a 128-token tile loses a CTA per SM and measured slower.

### Limitations

- Q2 end-to-end throughput is not reported: there is no local PQ2 artifact, and a 27B Q2 artifact
  would leave too little of the 8 GB budget for the runtime workspace and KV cache. Q2 is qualified
  at the Op level and against the independent oracle.
- Decode is instruction-bound at about 48 ms per token against a 17 ms memory floor. Further decode
  gains need fewer decode instructions or a different arithmetic tier, which would change the A16
  contract.
- All numbers are from the local RTX 5060 Laptop under Dynamic Boost. They are not RTX 5090
  results, and the tuning thresholds are not portable.

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
