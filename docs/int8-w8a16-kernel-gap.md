# XPU INT8 W8A16 Kernel Gap — Analysis & Fix Plan

> **Date**: 2026-08-16
> **Status**: DONE — model loads, generates correct text, benchmarked on 2x B70
> **Trigger**: `lued/Qwen3.8-27B-INT8-W8A16-MTP` fails to load on the B70 XPU stack

## TL;DR

The Intel XPU vLLM fork has no INT8 W8A16 (WNA16 with 8-bit weights) linear
kernel. Compressed-tensors INT8 checkpoints (the llm-compressor default W8A16
recipe) fail at model init with:

```
ValueError: Failed to find a kernel that can implement the WNA16 linear layer.
  XPUW4A8IntLinearKernel cannot implement due to: XPUW4A8Int requires int4 weights, got uint8b128
  XPUwNa16LinearKernel cannot implement due to: Quant type (uint8b128) not supported by XPUwNa16,
  supported types are: (ScalarType.uint4, ScalarType.uint4b8)
```

The fix requires a new oneDNN-backed `int8_gemm_w8a16` primitive in
`vllm-xpu-kernels` (C++/SYCL) plus a new `XPUw8a16IntLinearKernel` in the
Python kernel layer.

## Why it fails

### Kernel selection path

`compressed_tensors_wNa16.py` maps `num_bits=8` → `scalar_types.uint8b128`
(8-bit values packed 4-per-32-bit-word). It then calls
`choose_mp_linear_kernel()`, which walks `_POSSIBLE_KERNELS[PlatformEnum.XPU]`
in priority order:

| # | Kernel | Accepts weight_type | lued model has |
|---|--------|--------------------|----------------|
| 1 | `XPUW4A8IntLinearKernel` | `int4` only | `uint8b128` |
| 2 | `XPUwNa16LinearKernel` | `uint4`, `uint4b8` | `uint8b128` |

Neither matches → `ValueError` raised at `create_weights()` for the first
quantized linear (GDN `in_proj_qkvz`).

### What the XPU C++ kernel library contains

`vllm-xpu-kernels/csrc/xpu/onednn/` — five GEMM primitives, all oneDNN-backed:

| Primitive | Weights | Activations |
|-----------|---------|-------------|
| `fp8_gemm` | fp8 | fp8 (W8A8) |
| `fp8_gemm_w8a16` | fp8 | bf16/fp16 |
| `fp4_gemm` | fp4 | fp4 (W4A4) |
| `int4_gemm_w4a16` | int4 | bf16/fp16 |
| `int4_gemm_w4a8` | int4 | int8 (W4A8) |

**No int8 W8A16 primitive.** The `joint_dtypes_t` enum in `onednn_ext.h` has
`f16_int4`, `bf16_int4`, `s8_int4`, `u8_int4`, fp8/fp4 combos — but no
`bf16_int8` / `f16_int8`.

### The lued checkpoint layout

- `weight_packed`: `[N, K/4]` int32 — 4 int8 values per i32 (group-128 symmetric)
- `weight_scale`: `[N, K/128]` bf16
- `weight_shape`: `[2]` int64 (original `[N, K]`)
- No zero points (symmetric quantization)
- 400 quantized Linears: 192 MLP, 64 full-attention, 144 GDN projections
- BF16 preserved: vision tower, lm_head, MTP, GDN `in_proj_a`/`in_proj_b`

## Why the cheap alternatives don't work

| Option | Verdict |
|--------|---------|
| Dequant to BF16 at load | 30GB → 54GB = 27GB/GPU. Weights alone eat the whole 0.85 budget on 32GB cards. Zero KV room. Dead end. |
| Dequant to FP8, use `XPUW8A16FP8LinearKernel` | int8 → fp8_e4m3 is lossy (4-bit mantissa). Re-quantizing a quantized model adds a second error round. Defeats the purpose. |
| Run on RTX 5090 (CUDA) | Works today — Marlin handles INT8 W8A16 natively. But that's the Windows box, not the B70 server. |

## The fix (implemented)

Test tree: `vllm/vllm-src-int8-test/` — vLLM `83f591d` (main),
vllm-xpu-kernels `13013c5` (main). Test venv: `.venv-int8-test`
(torch `2.13.0+xpu`, triton-xpu `3.7.2`). Patches saved to
`/tmp/int8-w8a16-kernels.patch` (315 lines) and
`/tmp/int8-w8a16-vllm.patch` (109 lines).

### C++ side (vllm-xpu-kernels) — 5 files

1. **`onednn_ext.h`**: added `f16_int8` + `bf16_int8` to `joint_dtypes_t`;
   type mappers `(f16, s8, f16)` / `(bf16, s8, bf16)`; two `case` branches in
   the `matmul_primitive_create_and_cache` dispatch (before `default:`).
2. **New `int8_gemm_w8a16.h`**: `dnnl_matmul_w8a16_int8()`, modeled on
   `int4_gemm_w4a16.h` but simpler — no bit-unpacking (weights are 1 byte
   each), no zero points. Weight passed as `[k, n]` with k contiguous
   (`trans_type_t::nt`). Scales set with
   `pattr.set_scales(DNNL_ARG_WEIGHTS, mask (1<<0)+(1<<1), {group_size, 1}, bf16)`
   — group quant along k, per-channel along n. Also sets
   `set_fpmath_mode(f16/bf16, /*apply_to_int=*/true)` — required by oneDNN
   for integral weights with a K-dim scale mask (see below).
3. **`onednn_matmul.cpp`**: `int8_gemm_w8a16()` wrapper (dtype checks: B must
   be s8, A must be bf16/fp16).
4. **`torch_bindings.cpp`**: registered
   `int8_gemm_w8a16(Tensor A, Tensor B, Tensor? bias, Tensor B_scale, int group_size) -> Tensor`.
5. **`ops.h`**: declaration.

### Python side (vllm) — 2 files

6. **`mixed_precision/xpu.py`**: new `XPUw8a16IntLinearKernel(MPLinearKernel)`.
   - `can_implement`: `weight_type == uint8b128`, bf16/fp16 act, symmetric,
     no g_idx, group_size % 32 == 0, in/out % 32 == 0.
   - `process_weights_after_loading`: unpack `weight_packed`
     `[N, K/4]` int32 → `[N, K]` int8 via
     `(w_packed.view(torch.uint8).to(torch.int16) - 128).to(torch.int8)`
     (the `uint8b128` +128 offset convention; little-endian, 4 int8 per i32,
     no memory growth); transpose `weight_scale` `[N, K/128]` → `[K/128, N]`.
   - `apply_weights`: `torch.ops._xpu_C.int8_gemm_w8a16(reshaped_x, w_q.t(), bias, w_s, group_size)`.
7. **`linear/__init__.py`**: import + add to
   `_POSSIBLE_KERNELS[PlatformEnum.XPU]` (after `XPUwNa16LinearKernel`) + `__all__`.

### Key layout facts (verified from the checkpoint)

- `weight_packed` `[N, K/4]` int32, **little-endian** byte order (byte 0 =
  first int8 of the group). Example `in_proj_qkv`: `[10240, 1280]` → N=10240,
  K=5120.
- **Each byte stores `int8_value + 128`** (the `uint8b128` offset
  convention), NOT raw two's-complement int8. Unpacking is
  `w_packed.view(torch.uint8).to(torch.int16).sub_(128).to(torch.int8)`.
  A plain `.view(torch.uint8).view(torch.int8)` bitcast silently corrupts
  every byte >= 128 (verified: corr -0.59 vs base bf16 with the bitcast,
  corr +1.00000 with the offset subtracted).
- `weight_scale` `[N, K/128]` bf16, symmetric (no zero points). Scales match
  the base bf16 per-group absmax / 127 exactly — the quantization itself is
  faithful.
- The int4 kernel's scale pattern (`{group_size, 1}`) is the correct one for
  this layout — NOT the fp8 kernel's 2D block-quant pattern
  (`{blk_group_size, blk_group_size}`), which assumes a `[k/g, n/g]` scale.

### oneDNN fpmath requirement (found during bring-up)

oneDNN's bf16×s8 group-scaled matmul works on Xe2, but
`matmul_pd.hpp::attr_scales_ok` requires `fpmath_mode` with
`apply_to_int=true` for integral weights carrying a K-dimension scale mask
(group quant). Without it the descriptor is rejected with "unsupported
scales configuration". The int4 kernel already set this; the int8 kernel had
to add it too:

```cpp
pattr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
if (in_dtype == at::ScalarType::BFloat16)
  pattr.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
```

The engine that handles this descriptor is `jit:gemm:any`
(`src/gpu/intel/gemm/jit.hpp`, `wei_decomp()` accepts s8 weights).

## Results (2026-08-16)

Test venv `.venv-int8-test`: vLLM `83f591d`, vllm-xpu-kernels `13013c5` +
patch, torch `2.13.0+xpu`, oneAPI 2026.1.0, 2x Arc Pro B70 (TP=2).

- **Kernel**: standalone repro (`repro-int8-gemm.py`) passes; numerical check
  vs CPU dequant reference on real checkpoint weights: corr 0.999996,
  mean rel diff 1.0% (bf16 rounding).
- **Model**: `lued-Qwen3.8-27B-INT8-W8A16-MTP` loads (14.62 GiB/GPU),
  generates correct text. Smoke test "17 * 23" → `\n\n391` + EOS, identical
  to the base bf16 model (TP=4) and the INT4 AutoRound model (TP=2).
- **Decode throughput** (512 tokens, chat template, enforce_eager, TP=2,
  same venv, same prompt):

  | Model | Total tok/s (incl. prefill) |
  |-------|-----------------------------|
  | INT8 W8A16 (lued) | **10.30** |
  | INT4 AutoRound (w4g128) | 9.23 |

  INT8 is ~12% faster than INT4 under identical conditions. Both are far
  below the 48 tok/s INT4 TP=4 number from the main venv (0.26.1rc1,
  cudagraph enabled) — the test venv is bleeding-edge dev with
  enforce_eager, so treat this as a relative comparison only. Note: the
  custom op has no FakeTensor registration, so the torch.compile / cudagraph
  path fails at memory profiling — enforce_eager is required until a fake
  impl is added.
- **Two bugs found during bring-up** (both fixed, both in the patch):
  1. oneDNN `fpmath_mode` missing (C++ kernel) — descriptor rejected.
  2. `uint8b128` +128 offset not applied when unpacking (Python kernel) —
     model looped on a single byte-level BPE token, empty output.

## Why INT8 W8A16 matters

- INT8 W8A16 is the llm-compressor default W8A16 recipe — a common format
- lued's KLD: 0.000894 nats/token (99.36% fidelity) vs official FP8's
  0.004396 (98.53%) — measurably better than FP8
- 30GB checkpoint fits TP=2 on B70s with room for 256k KV
- Benefits any future INT8 W8A16 model on XPU, not just this one
