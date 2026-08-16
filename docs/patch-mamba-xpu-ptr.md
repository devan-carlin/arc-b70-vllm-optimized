# Patch: Mamba XPU pointer fix (`patches/mamba-xpu-ptr.patch`)

Target: `vllm` (Python), `vllm/v1/worker/mamba_utils.py`. 1 file, ~43 lines.

## Problem

On XPU, `tensor.data_ptr()` returns device addresses in a high
virtual-address region whose top bit is set (e.g.
`0xffffe000ff800000`) — a value larger than `int64 max`.

The mamba/GDN state bookkeeping stores these pointers into int64 tensors
(`state_base_addrs`, `block_table_ptrs`) for the fused Triton kernels.
Assigning an unsigned 64-bit value above `int64 max` to an int64 tensor
element raises:

```
ValueError: Overflow when unpacking long long
```

This breaks any model that uses mamba-style linear attention on XPU —
including Qwen3.8-27B (48 GDN linear-attention layers + 16 full-attention
layers).

## What the patch does

Adds a helper and applies it at the two pointer-storage sites:

```python
def _ptr_to_i64(ptr: int) -> int:
    """Reinterpret an unsigned 64-bit device pointer as signed int64."""
    if ptr >= (1 << 63):
        ptr -= 1 << 64
    return ptr
```

Call sites (in `MambaState` init / block-table setup):

```python
# before
self.state_base_addrs[idx] = state.data_ptr()
self.block_table_ptrs[i] = bt.data_ptr()

# after
self.state_base_addrs[idx] = _ptr_to_i64(state.data_ptr())
self.block_table_ptrs[i] = _ptr_to_i64(bt.data_ptr())
```

## Why this is safe

The fused mamba kernels only ever **bitcast** these int64 values back to
pointers inside Triton (`.to(tl.pointer_type(...))`). They never do signed
arithmetic on them. Two's-complement reinterpretation is therefore
lossless: the stored signed int64 round-trips to the exact original
64-bit address.

## Apply

```bash
cd vllm
git apply ../patches/mamba-xpu-ptr.patch
```

No rebuild needed.

## Important: venv-local, lost on rebuild

This patch is applied to the **installed** vllm inside the venv
(`.venv/lib/python3.x/site-packages/vllm/v1/worker/mamba_utils.py`), not to
a source tree. A `pip install` / venv rebuild silently drops it. If a
mamba/GDN model starts failing with the overflow error after a rebuild,
re-apply this patch.

The INT8 W8A16 test model does not exercise the MTP path but **does** use
GDN linear attention, so the test venv needs this patch too if it hits the
overflow.

## Upstream

Candidate for an upstream vLLM PR — the fix is device-agnostic in spirit
(any backend with high-bit-set device pointers would hit this), though it
has only been observed on XPU.
