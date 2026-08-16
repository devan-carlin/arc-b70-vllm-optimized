# Patch: INT8 W8A16 kernels (`patches/int8-w8a16-kernels.patch`)

Target: `vllm-xpu-kernels` (C++/SYCL, oneDNN backend). 5 files, ~295 lines.

## Problem

vLLM's XPU path had no W8A16 (int8 weights, bf16/fp16 activations) linear
kernel. The `matmul_primitive_create_and_cache` dispatch in
`csrc/xpu/onednn/onednn_ext.h` only handled int4 and fp8 GEMM; anything else
hit `default: throw "Only support int4 and fp8 gemm ..."`.

This blocked loading INT8 W8A16 checkpoints (e.g. lued's
`Qwen3.8-27B-INT8-W8A16-MTP`, CompressedTensors WNA16 with 8-bit weights) on
XPU. The model's linear layers fell through with no kernel able to
`can_implement`.

## Why oneDNN can do this

oneDNN 3.13 (pinned via CMake FetchContent, tag `rls-v3.13`) supports
bf16/f16 source x s8 weight matmul with per-group weight scales. Verified in
`src/gpu/intel/matmul/ref.hpp`. The int4 W4A16 kernel in the same tree is the
proven analog — this patch mirrors its structure with s8 weights (no packing,
no zero point).

## What the patch does

### 1. `csrc/xpu/onednn/int8_gemm_w8a16.h` (new, ~133 lines)

`dnnl_matmul_w8a16_int8(result, mat1, mat2, bias, m2_sc, group_size)`:

- `mat1` (src): `[b, m, k]` bf16/f16
- `mat2` (weights): `[k, n]` s8, k contiguous (passed as `.t()` of a
  contiguous `[n, k]` tensor, `trans_type_t::nt`)
- `m2_sc` (scales): `[k/group_size, n]` bf16 — group quant along k,
  per-channel along n
- `result` (dst): `[b, m, n]` same dtype as src

Key details:

- Joint dtype selected from the activation dtype: `f16_int8` or
  `bf16_int8`.
- `ldb` computed from `mat2` strides: if the last dim is contiguous
  (stride 1), use the second-to-last stride, else the last.
- Scale attribute (the part that is easy to get wrong):

  ```cpp
  pattr.set_scales(
      DNNL_ARG_WEIGHTS,
      /* mask */ (1 << 0) + (1 << 1),
      {group_size, 1},
      get_onednn_dtype(m2_sc));
  ```

  This is the **int4 kernel's 1D group-quant pattern** (`{group_size, 1}`),
  NOT the fp8 kernel's 2D block-quant pattern (`{blk, blk}` with an encoded
  `sc_group_size`). The lued checkpoint uses 1D group quant along K plus
  per-channel along N; using the fp8 pattern produces silently wrong
  results. This was caught in review before the first build.
- **`fpmath_mode` (required, found during bring-up)**: oneDNN's
  `matmul_pd.hpp::attr_scales_ok` rejects integral (s8) weights carrying a
  K-dimension scale mask unless `attr()->fpmath_.apply_to_int_` is true.
  Without it the descriptor fails with "unsupported scales configuration".
  The int4 kernel already set this; the int8 kernel must too:

  ```cpp
  pattr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
  if (in_dtype == at::ScalarType::BFloat16)
    pattr.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
  ```

  The engine that handles the resulting descriptor is `jit:gemm:any`
  (`src/gpu/intel/gemm/jit.hpp`, `wei_decomp()` accepts s8 weights).
- User scratchpad mode, scale passed as a primitive attribute, then
  SRC/WEIGHTS/DST/BIAS/SCRATCHPAD arg handles and execute.

### 2. `csrc/xpu/onednn/onednn_ext.h`

- `joint_dtypes_t`: add `f16_int8`, `bf16_int8`.
- `onednn_types_mapper` specializations:
  - `f16_int8` -> `(f16, s8, f16)`
  - `bf16_int8` -> `(bf16, s8, bf16)`
- `matmul_primitive_create_and_cache` dispatch: two new `case` branches
  (before the `default` throw) forwarding to the templated
  `matmul_primitive_create_and_cache<joint_dtypes_t::f16_int8/bf16_int8, F>`.

### 3. `csrc/xpu/onednn/onednn_matmul.cpp`

`int8_gemm_w8a16(A, B, bias, B_scale, group_size)` wrapper:

- `B_scale` must be contiguous
- `B` must be `at::ScalarType::Char` (int8)
- `A` must be bf16 or fp16
- `check_and_create_output_tensor(A, B, A.scalar_type())` then call
  `oneDNN::dnnl_matmul_w8a16_int8`.

### 4. `csrc/xpu/ops.h`

Declaration of `int8_gemm_w8a16`.

### 5. `csrc/xpu/torch_bindings.cpp`

```cpp
xpu_ops.def(
    "int8_gemm_w8a16(Tensor A, Tensor B, Tensor? bias, Tensor B_scale, "
    "int group_size) -> Tensor");
xpu_ops.impl("int8_gemm_w8a16", torch::kXPU, &int8_gemm_w8a16);
```

## Weight layout (verified against the checkpoint)

- `weight_packed`: `[N, K/4]` int32, **little-endian** byte order (byte 0 =
  first int8 of the group). **Each byte stores `int8_value + 128`** (the
  `uint8b128` offset convention) — unpacking is
  `view(torch.uint8).to(torch.int16).sub_(128).to(torch.int8)`, NOT a plain
  two's-complement bitcast. Verified against the base bf16 weights: offset
  unpack gives corr +1.00000, bitcast gives -0.59 (see the Python patch doc
  for the full story).
- `weight_scale`: `[N, K/128]` bf16 (group_size 128). Scales match the base
  bf16 per-group absmax / 127 exactly.
- Symmetric: no zero points.

Example, `in_proj_qkv`: packed `(10240, 1280)` -> N=10240, K=5120;
scale `(10240, 40)`.

## Apply

```bash
cd vllm-xpu-kernels
git apply ../patches/int8-w8a16-kernels.patch
```

Requires a full kernel rebuild (oneDNN + CUTLASS-SYCL tree, ~1.5-2 h with
`VLLM_XPU_KERNELS_MAX_JOBS=4`).

## Verify

After build, the op must be present:

```python
import vllm_xpu_kernels._C as C
print([n for n in dir(C) if "int8" in n])  # expect int8_gemm_w8a16
```

## Verification results (2026-08-16)

- Standalone repro (`repro-int8-gemm.py`, m=1 k=5120 n=10240 g=128):
  descriptor created via `jit:gemm:any` with `attr-fpmath:bf16:true`, exec
  clean.
- Numerical check vs CPU dequant reference on real checkpoint weights
  (`verify-int8-real-weights.py`): corr 0.999996, mean rel diff 1.0% (bf16
  rounding).
- Full model: `lued-Qwen3.8-27B-INT8-W8A16-MTP` TP=2 on 2x Arc Pro B70
  generates correct text ("17 * 23" -> `\n\n391` + EOS, matching base bf16
  and INT4 AutoRound).
- Decode: 10.30 tok/s (512 tokens, enforce_eager, TP=2) vs 9.23 tok/s for
  INT4 AutoRound under identical conditions.

## Known limitations

- No FakeTensor (fake) registration for `int8_gemm_w8a16` — the
  torch.compile / cudagraph path fails at memory profiling with
  `UnsupportedOperatorException`. `enforce_eager=True` is required until a
  fake impl is added (the int4 op has the same gap).

## Related

- Python side: `docs/patch-int8-w8a16-vllm.md`
- Full analysis: `docs/int8-w8a16-kernel-gap.md`
