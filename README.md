# Intel Arc Pro B70: Optimized vLLM (oneAPI 2026.1 + Triton XPU)

Build, patch, and serve LLMs on 4x Intel Arc Pro B70 (Battlemage, Xe2, 32 GB
each) with vLLM on the Intel XPU stack. This repo is self-contained: a build
script, a tuned launch script, the patches, and the performance notes.

Tested on: Ubuntu 26.04, oneAPI 2026.1, torch 2.13.0+xpu, triton-xpu 3.7.2,
vLLM main (XPU), vllm-xpu-kernels main.

## What's in here

| Path | What it is |
|------|-----------|
| `scripts/build-vllm-xpu.sh` | End-to-end build: torch XPU, vllm + vllm-xpu-kernels from source, patches applied, triton-xpu enforced |
| `scripts/serve.sh` | Launch script with the tuned B70 flags (XPU graph, fp8 KV, block-size 32, batched-tokens 16384, ...) |
| `scripts/setup-gpu-power-limits.sh` | Power/frequency caps via sysfs + systemd persistence (160W sweet spot) |
| `patches/int8-w8a16-*.patch` | New INT8 W8A16 linear kernel (C++/SYCL + Python) — see `docs/int8-w8a16-kernel-gap.md` |
| `patches/moe-topk-16-32-64.patch` | MoE TopK=16/32/64 support in the XPU MoE kernels |
| `patches/mamba-xpu-ptr.patch` | Fix XPU pointer overflow in the mamba/GDN state bookkeeping |
| `docs/int8-w8a16-kernel-gap.md` | Full analysis of the INT8 W8A16 kernel gap + the fix |
| `docs/patch-*.md` | Per-patch analysis + implementation notes (one doc per patch) |

## Quick start

```bash
# 1. Prereqs: oneAPI 2026.1 (compiler + DNNL + CCL), xe driver, python3.12
#    (see the oneAPI installer; the build script checks for setvars.sh)

# 2. Build (creates $HOME/vllm-xpu/.venv, clones + compiles, applies patches)
bash scripts/build-vllm-xpu.sh

# 3. Serve
bash scripts/serve.sh /path/to/your-model
```

The build is the long pole: vllm-xpu-kernels compiles the oneDNN +
CUTLASS-SYCL tree, which takes well over an hour on a 12-core CPU. That is
normal.

## The tuned launch flags

These are the flags that matter on B70, all set by `serve.sh`:

| Flag | Value | Why |
|------|-------|-----|
| `VLLM_XPU_ENABLE_XPU_GRAPH=1` | env | XPU graph capture for decode — the single biggest decode win |
| `--kv-cache-dtype fp8` | flag | Halves KV cache memory, enabling 256k context on 32 GB cards |
| `--block-size 32` | flag | Larger KV blocks, better memory access on XPU |
| `--max-num-batched-tokens 16384` | flag | Bigger prefill batches — the main prefill throughput lever |
| `--max-num-seqs 2` | flag | Single-user box: cap sequences, keep memory for KV |
| `--language-model-only` | flag | Skip the vision tower for VLM checkpoints (Qwen3.x) |
| `--gpu-memory-utilization 0.85` | flag | Leave headroom for the XPU runtime |

The XPU environment block in `serve.sh` (device selector, CCL IPC, graph
comm capture) is required for multi-GPU tensor parallelism to work at all.

## Patches

### INT8 W8A16 linear kernel (`patches/int8-w8a16-*.patch`)

The XPU kernel library had no INT8 W8A16 (WNA16, 8-bit weights) linear
kernel, so compressed-tensors INT8 checkpoints — the llm-compressor default
W8A16 recipe — failed at model init. This adds a oneDNN-backed
`int8_gemm_w8a16` primitive (bf16/fp16 activations x s8 weights, symmetric
group-quant) plus the `XPUw8a16IntLinearKernel` Python wrapper. oneDNN 3.13
supports bf16 x s8 matmul on Xe2; the kernel just wasn't wired up. Full
analysis in `docs/int8-w8a16-kernel-gap.md`; per-patch details in
`docs/patch-int8-w8a16-kernels.md` (C++) and `docs/patch-int8-w8a16-vllm.md`
(Python).

Apply to a vllm + vllm-xpu-kernels source tree:

```bash
cd vllm-xpu-kernels && git apply ../patches/int8-w8a16-kernels.patch
cd ../vllm && git apply ../patches/int8-w8a16-vllm.patch
# then rebuild both (see build-vllm-xpu.sh)
```

### MoE TopK 16/32/64 (`patches/moe-topk-16-32-64.patch`)

The XPU MoE kernels only dispatched TopK=8. Qwen3.6-35B-A3B and similar MoE
models can run at TopK=16/32/64 (more active experts, better quality). This
adds the missing dispatch cases to `remap_hidden_states` and `moe_gather`.
Details and benchmark numbers in `docs/patch-moe-topk-16-32-64.md`.

### Mamba XPU pointer fix (`patches/mamba-xpu-ptr.patch`)

On XPU, `tensor.data_ptr()` returns addresses in a high virtual-address
region whose top bit is set — larger than int64 max. Storing them in an
int64 tensor raised `Overflow when unpacking long long`, breaking any model
with mamba/GDN linear-attention layers (Qwen3.5/3.6/3.8 hybrid attention).
The fix reinterprets the pointer as two's-complement signed int64, which is
lossless because the kernels only ever bitcast it back to a pointer. Details
in `docs/patch-mamba-xpu-ptr.md`.

## Power tuning

LLM inference is memory-bandwidth bound, so underclocking costs little:

| Setting | Power/GPU | 27B Q4 throughput |
|---------|-----------|-------------------|
| Stock | ~175W | ~12-15 tok/s |
| **160W cap (default)** | ~160W | ~11-14 tok/s (~5-10% slower) |
| 140W cap | ~140W | ~10-13 tok/s (~15% slower) |

```bash
sudo bash scripts/setup-gpu-power-limits.sh            # 160W, persistent
sudo bash scripts/setup-gpu-power-limits.sh --status   # show current state
sudo bash scripts/setup-gpu-power-limits.sh --reset    # remove limits
```

## Performance notes

- **MTP (multi-token prediction) speculative decoding**: do not use it for
  dense Qwen 27B-class models on XPU. Measured 34.9 tok/s with MTP vs
  52.2 tok/s without. The draft-accept overhead exceeds the win at these
  batch sizes. (MoE models may differ — the 35B-A3B gist numbers use MTP.)
- **Tensor parallelism**: TP=2 on a 27B dense model is the sweet spot for
  256k context (weights ~15 GB/GPU, room for fp8 KV). TP=4 is fine for
  shorter contexts and lower per-GPU memory.
- **Prefill**: with `--max-num-batched-tokens 16384`, expect 2k+ tok/s
  prefill on a 27B dense INT4 model at TP=1-2. MoE models (35B-A3B) clear
  11k+ tok/s prefill at TP=1.
- **Decode**: ~50-90 tok/s single-stream depending on model size and
  quantization, with XPU graph enabled.

## Benchmarking

```bash
uvx llama-benchy --base-url http://0.0.0.0:8000/v1 \
  --model <served-name> --pp 4096 --tg 256 --concurrency 1 2 \
  --no-cache --exact-tg --latency-mode generation
```

## Notes

- The build script pins nothing by default — it builds `main` of both repos.
  If a future vLLM main breaks the XPU build, pin a known-good commit in the
  clone step.
- `vllm-gguf-plugin` is installed for GGUF models; not needed for
  safetensors checkpoints.
- This repo is the operational "how". The research notes, experiments, and
  model library live in the companion `electric-sheep` repo.
