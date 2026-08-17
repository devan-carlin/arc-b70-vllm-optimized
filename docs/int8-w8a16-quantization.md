# INT8 W8A16 Quantization (making the checkpoint)

**Companion to** [`int8-w8a16-kernel-gap.md`](./int8-w8a16-kernel-gap.md) (running
INT8) and the two `int8-w8a16-*.patch` files (the kernel). This doc covers
**producing** a checkpoint the patched path can load.

**Script:** `scripts/quantize-w8a16.py`
**Verified on:** `Qwen3.8-27B` (qwen3_5 arch, 27B, 52 GB bf16) → 31.6 GB INT8,
47.8 tok/s decode at TP=4 with XPU graph (matches the INT4 AutoRound baseline).

---

## Why RTN

| Option | Why not (for a one-off local model) |
|--------|-------------------------------------|
| AutoRound INT8 | Its INT8 export writes scales as **E8M0-encoded uint8**, not plain bf16. The XPU kernel expects bf16 `weight_scale`. Needs a decode fix on top. |
| llm-compressor W8A16 | Heavier dependency + calibration pass. Same `pack-quantized` layout, but more moving parts. |
| **RTN (this script)** | Writes bf16 scales directly → format-compatible out of the box. No calibration. ~3 min for a 27B model. Round-trip dequant rel err 0.004 (= 0.5/127, the RTN floor). |

RTN is the right tool when the goal is a working, fast checkpoint rather than a
quality-maximized one. AutoRound is the upgrade path if you later want better
quality (fix the E8M0 scale encoding first).

## Checkpoint format (what the kernel consumes)

For each quantized 2D linear `[N, K]` (bf16):

| Tensor | dtype | shape | meaning |
|--------|-------|-------|---------|
| `weight_packed` | int32 | `[N, K/4]` | 4 int8 per int32, little-endian. Each byte = `int8_value + 128` (`uint8b128` offset, **not** two's complement). |
| `weight_scale` | bf16 | `[N, K/128]` | Per-group (group_size=128) `absmax / 127`. |

8-bit symmetric → compressed-tensors scalar type `uint8b128`. No zero points.

The `+128` offset is the key byte-level fact: the C++ kernel treats unpacked
values as signed int8, so the Python side must do
`(packed.view(uint8).to(int16) - 128).to(int8)` — a plain `.view(int8)`
reinterpret is wrong for any byte ≥ 128 (see the kernel-gap doc).

## Usage

```bash
python scripts/quantize-w8a16.py <src_bf16_dir> <dst_int8_dir> [--group-size 128]
```

The script:
- Copies all non-safetensors files (config, tokenizer, ...).
- Replaces each target `.weight` with `.weight_packed` + `.weight_scale`.
- Rebuilds `model.safetensors.index.json` (metadata values stringified —
  safetensors requires str→str metadata).
- Writes the `quantization_config` into `config.json` (see below).

## The three config gotchas (qwen3_5 multimodal)

These are what make a qwen3_5 checkpoint actually **load**. All three are
handled by the script; they are documented here because each one produces a
different, confusing failure if you hand-write the config.

### 1. `quant_method: compressed-tensors` is mandatory

Without it, vLLM builds every layer as plain bf16 (no `weight_packed`), and
loading the packed tensors crashes with:
```
AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'
```

### 2. The ignore list matches the *construction* prefix, not the weight name

`should_ignore_layer` runs at layer-build time against the construction
prefix. For the qwen3_5 multimodal wrapper the language linears are built under
`language_model.model.layers.N.<proj>` (top-level prefix `""`), **not**
`model.layers.N.<proj>`. A naive `re:^(?!model\.layers\.)` ignores the whole
model. The script writes:
```json
"re:^(?!.*language_model\\.model\\.layers\\.)"
```
(the `.*` makes it robust to any leading prefix). For a non-multimodal model
the prefix is usually just `model.layers.N.*`.

### 3. Fused-layer ignore checks the *unfused* shard names

The GDN `in_proj_ba` layer merges `in_proj_a` + `in_proj_b` (N=48 each). N=48
is never a multiple of 32 at any TP, so the kernel's `in/out % 32 == 0`
constraint rejects it. For a fused layer, `should_ignore_layer` checks the
**unfused** shard names, so `re:.*in_proj_ba` does not work — the rule must
match the shards:
```json
"re:.*in_proj_[ab]$"
```
Coupled requirement: `in_proj_a`/`in_proj_b` are excluded from the quantizer's
`LINEAR_SUFFIXES`, so their source weights stay bf16 in the checkpoint (a bf16
layer must load bf16 tensors).

**Final ignore list:**
```json
"ignore": [
  "re:^(?!.*language_model\\.model\\.layers\\.)",
  "re:.*in_proj_[ab]$"
]
```

## Cudagraph requirement

`enforce_eager=False` alone is not enough on XPU. You also need the XPU graph
env vars (set by `scripts/serve.sh`):
```bash
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
```
plus the `_xpu_ops.py` FakeTensor registration from
`patches/int8-w8a16-vllm.patch`. Together these are the difference between
19.6 and 47.8 tok/s at TP=4.

## Results

| Config | tok/s |
|--------|-------|
| INT8 W8A16, TP=4, XPU graph ON | **47.80** |
| INT8 W8A16, TP=4, XPU graph OFF | 19.58 |
| INT4 AutoRound, TP=4 (baseline) | 48.0 |

Quantization is not the bottleneck; the environment (graph, TP) is.

## Validation

- Correctness: prompt "What is 17 * 23?" → `'\n\n391'` (TP=2 and TP=4).
- Quantizer self-test: round-trip dequant rel err 0.004; byte convention and
  per-group `absmax/127` scales match a known-good checkpoint.

## Related

- Kernel (running INT8): [`int8-w8a16-kernel-gap.md`](./int8-w8a16-kernel-gap.md)
- C++ patch: [`patch-int8-w8a16-kernels.md`](./patch-int8-w8a16-kernels.md)
- Python patch (incl. FakeTensor): [`patch-int8-w8a16-vllm.md`](./patch-int8-w8a16-vllm.md)
