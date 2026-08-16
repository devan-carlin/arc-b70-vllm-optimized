# XPU INT8 W8A16 Kernel Gap — Analysis & Fix Plan

> **Date**: 2026-08-16
> **Status**: Implemented in a clean test venv (bleeding-edge components); build + model test in progress
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
   — group quant along k, per-channel along n.
3. **`onednn_matmul.cpp`**: `int8_gemm_w8a16()` wrapper (dtype checks: B must
   be s8, A must be bf16/fp16).
4. **`torch_bindings.cpp`**: registered
   `int8_gemm_w8a16(Tensor A, Tensor B, Tensor? bias, Tensor B_scale, int group_size) -> Tensor`.
5. **`ops.h`**: declaration.

### Python side (vllm) — 2 files

6. **`mixed_precision/xpu.py`**: new `XPUw8a16IntLinearKernel(MPLinearKernel)`.
   - `can_implement`: `weight_type == uint8b128`, bf16/fp16 act, symmetric,
     no g_idx, group_size % 32 == 0, in/out % 32 == 0.
   - `process_weights_after_loading`: bitcast `weight_packed`
     `[N, K/4]` int32 → `[N, K]` int8 via `.view(torch.uint8).view(torch.int8)`
     (little-endian, 4 int8 per i32 — no memory growth); transpose
     `weight_scale` `[N, K/128]` → `[K/128, N]`.
   - `apply_weights`: `torch.ops._xpu_C.int8_gemm_w8a16(reshaped_x, w_q.t(), bias, w_s, group_size)`.
7. **`linear/__init__.py`**: import + add to
   `_POSSIBLE_KERNELS[PlatformEnum.XPU]` (after `XPUwNa16LinearKernel`) + `__all__`.

### Key layout facts (verified from the checkpoint)

- `weight_packed` `[N, K/4]` int32, **little-endian** byte order (byte 0 =
  first int8 of the group). Example `in_proj_qkv`: `[10240, 1280]` → N=10240,
  K=5120.
- `weight_scale` `[N, K/128]` bf16, symmetric (no zero points).
- The int4 kernel's scale pattern (`{group_size, 1}`) is the correct one for
  this layout — NOT the fp8 kernel's 2D block-quant pattern
  (`{blk_group_size, blk_group_size}`), which assumes a `[k/g, n/g]` scale.

### Risk

oneDNN's bf16×s8 group-scaled matmul on Xe2 (Battlemage) is a standard
primitive (confirmed in oneDNN 3.13 `src/gpu/intel/matmul/ref.hpp`:
`is_bf16 = src_dt_==bf16 && one_of(wei_dt_, bf16,s8,u8,s4,u4)`), but is
untested in this codebase. The int4 path's `fpmath_mode` tricks were not
carried over (not needed for s8 weights).

## Test plan

- Clean venv, bleeding-edge vLLM XPU fork + oneDPI/oneDNN + SYCL
- Build vllm-xpu-kernels with the patch
- Load `lued-Qwen3.8-27B-INT8-W8A16-MTP` TP=2 on GPUs 0,1
- Smoke test: 17×23=391, then a longer generation
- Compare decode throughput vs the INT4 AutoRound model (52.2 tok/s baseline)

## Why INT8 W8A16 matters

- INT8 W8A16 is the llm-compressor default W8A16 recipe — a common format
- lued's KLD: 0.000894 nats/token (99.36% fidelity) vs official FP8's
  0.004396 (98.53%) — measurably better than FP8
- 30GB checkpoint fits TP=2 on B70s with room for 256k KV
- Benefits any future INT8 W8A16 model on XPU, not just this one
