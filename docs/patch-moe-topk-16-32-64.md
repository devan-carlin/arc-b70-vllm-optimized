# Patch: MoE TopK 16/32/64 (`patches/moe-topk-16-32-64.patch`)

Target: `vllm-xpu-kernels` (C++/SYCL MoE kernels). 2 files, 2 commits
(`828a80a`, `d0dc965`), ~73 lines.

## Problem

The XPU MoE kernels had hardcoded TopK dispatch tables limited to
`[1, 2, 4, 6, 7, 8, 10]`. Any MoE model configured with
`num_experts_per_tok` above 10 (e.g. 16, 32, 64) hit:

- `remap_hidden_states`: `throw std::runtime_error("Unsupported TopK value")`
- `moe_gather`: `TORCH_CHECK(false, "error: not support TOPK=...")`

This blocked experimenting with higher expert-activation counts on
Qwen3.6-35B-A3B-class MoE models (stock config is TopK=8).

## What the patch does

Two dispatch tables, one per kernel, extended identically:

### 1. `csrc/moe/remap_hidden_states.cpp` (commit `d0dc965`)

`DISPATCH_TOPK_LAUNCH` macro gains branches:

```cpp
} else if (TopK == 16) {
  LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 16);
} else if (TopK == 32) {
  LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 32);
} else if (TopK == 64) {
  LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 64);
}
```

### 2. `csrc/moe/moe_gather.cpp` (commit `828a80a`)

`MoeGatherLauncher` `CASE_TOPK` table gains:

```cpp
CASE_TOPK(16, ElemsPerItem)
CASE_TOPK(32, ElemsPerItem)
CASE_TOPK(64, ElemsPerItem)
```

Both kernels must support the same TopK set — `remap_hidden_states` runs
first (scatter into expert slots) and `moe_gather` runs last (aggregate
expert outputs back). Patching only one leaves the pipeline broken at the
other end.

## Apply

```bash
cd vllm-xpu-kernels
git am ../patches/moe-topk-16-32-64.patch   # it's a format-patch series
# or: git apply (applies both diffs, loses commit metadata)
```

Requires a kernel rebuild.

## Benchmark context (35B-A3B int4, B70, from electric-sheep)

Decode tok/s by TopK (concurrency 1-2 range):

| TopK | tok/s |
|------|-------|
| 8    | ~77-83 |
| 16   | ~70-79 |
| 32   | ~62-73 |
| 64   | ~56-62 |

More active experts = more weight traffic per token, so decode speed drops
monotonically. TopK=8 remains the speed default; 16/32/64 exist for
quality-vs-speed experiments (more experts per token can improve output
quality on some workloads).

## Notes

- TopK=16 was already observed working with stock vllm-xpu-kernels 0.1.13.1
  in some paths; the patch makes 16/32/64 explicit and uniform across both
  kernels.
- The patch also carries a file-mode change (100644 -> 100755) on the two
  .cpp files from the original commits; harmless.
