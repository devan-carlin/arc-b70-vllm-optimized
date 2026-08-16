# Patch: INT8 W8A16 vLLM kernel registration (`patches/int8-w8a16-vllm.patch`)

Target: `vllm` (Python). 2 files, ~109 lines.

## Problem

Even with the C++ `int8_gemm_w8a16` op available, vLLM's mixed-precision
linear kernel selection had no XPU kernel that could claim 8-bit
(`uint8b128`) weights. `choose_mp_linear_kernel` walks
`_POSSIBLE_KERNELS[PlatformEnum.XPU]` in order and picks the first kernel
whose `can_implement` returns True; for W8A16 checkpoints every entry
declined, so model load failed.

## What the patch does

### 1. `vllm/model_executor/kernels/linear/mixed_precision/xpu.py`

New `XPUw8a16IntLinearKernel(MPLinearKernel)`:

**`can_implement`** — accepts the layer only when all hold:

- platform is XPU
- activation dtype is bf16 or fp16
- `weight_type == scalar_types.uint8b128` (the CompressedTensors WNA16
  8-bit type; `WNA16_SUPPORTED_TYPES_MAP[8] = uint8b128`,
  `pack_factor = Fraction(32, 8) = 4`)
- symmetric quantization (no `zero_points`)
- no activation reordering (`has_g_idx` is False)
- `group_size % 32 == 0` (or -1)
- in/out sizes are multiples of 32

**`process_weights_after_loading`** — one-time transform after load:

- `weight_packed` is `[N, K/4]` int32 (4 int8 per i32, little-endian).
  **Each byte stores `int8_value + 128`** (the `uint8b128` offset
  convention), so unpacking subtracts the offset — a plain two's-complement
  bitcast is wrong for every byte >= 128:

  ```python
  w_u8 = w_packed.view(torch.uint8)  # [N, K] uint8, each = int8_value + 128
  w_i8 = (w_u8.to(torch.int16) - 128).to(torch.int8)  # [N, K] int8
  layer.weight_packed.data = w_i8
  ```

  No memory growth. Kept n-major so that `.t()` in `apply_weights` yields
  `[K, N]` with k contiguous (what the C++ kernel expects for
  `trans_type_t::nt`).

  **Why this matters (found during bring-up)**: with the bitcast, the model
  loaded and ran but looped on a single byte-level BPE token (empty output).
  The scales matched the base bf16 per-group absmax exactly, proving the
  quantization was faithful — only the value unpacking was wrong. Verified
  against the base bf16 weights: offset unpack corr +1.00000, bitcast
  corr -0.59.
- `weight_scale` is `[N, K/group_size]` -> transposed to
  `[K/group_size, N]` contiguous (the layout the C++ scale attribute
  expects).

**`apply_weights`**:

```python
reshaped_x = x.reshape(-1, x.shape[-1])          # [M, K]
out = torch.ops._xpu_C.int8_gemm_w8a16(
    reshaped_x,
    w_q.t(),                                     # [K, N] int8, k contiguous
    bias,
    w_s,                                         # [K/group_size, N] bf16
    self.config.group_size,
)
```

### 2. `vllm/model_executor/kernels/linear/__init__.py`

- Import `XPUw8a16IntLinearKernel`.
- Add it to `_POSSIBLE_KERNELS[PlatformEnum.XPU]` **after**
  `XPUwNa16LinearKernel` (order matters: W4A8 and WNA16 keep priority for
  their own weight types; this kernel only claims `uint8b128`).
- Add to `__all__`.

## Apply

```bash
cd vllm
git apply ../patches/int8-w8a16-vllm.patch
```

No rebuild needed for the Python side; the vllm package just needs to be
(re)installed or the tree used in place.

## Verify

On model load, the kernel selection log should show:

```
Using XPUw8a16IntLinearKernel for CompressedTensorsWNA16
```

for each W8A16 linear layer.

## Notes

- The kernel is deliberately conservative: asymmetric quant, g_idx
  reordering, and odd group sizes are rejected with a reason string so the
  selector can fall through (and fail loudly if nothing else claims the
  layer).
- `get_min_capability` returns -1 (no device capability gate on XPU).

## Related

- C++ side: `docs/patch-int8-w8a16-kernels.md`
- Full analysis: `docs/int8-w8a16-kernel-gap.md`
