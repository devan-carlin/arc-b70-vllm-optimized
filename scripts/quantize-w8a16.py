#!/usr/bin/env python3
"""RTN W8A16 (group-128, symmetric) quantizer for the XPU int8_gemm_w8a16 kernel.

Produces a compressed-tensors "pack-quantized" checkpoint that the patched
vLLM XPU path loads directly:
  - weight_packed : int32 [N, K/4]   (4 x uint8 per int32, LE; byte = int8 + 128)
  - weight_scale  : bf16  [N, K/128] (per-group absmax / 127)

Only 2D linears under `model.language_model.layers.*` are quantized. Visual,
MTP, norms, conv1d, embeddings and lm_head stay bf16 (excluded via the
quantization_config ignore list, which matches layer names).

Usage:
  python quantize-w8a16.py <src_dir> <dst_dir> [--group-size 128]
"""
import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# 2D linear weight suffixes under the language model that we quantize.
# NOTE: in_proj_a / in_proj_b are EXCLUDED on purpose. They merge into the
# GDN `in_proj_ba` layer (N=48), which is never a multiple of 32 at any TP, so
# the int8_gemm_w8a16 kernel cannot implement it. That layer is kept bf16 via
# the quantization_config ignore list, so its source weights must stay bf16
# here (not packed).
LINEAR_SUFFIXES = (
    "in_proj_qkv", "in_proj_z", "out_proj",
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
LANG_PREFIX = "model.language_model.layers."


def is_quant_target(name: str) -> bool:
    if not name.endswith(".weight"):
        return False
    if not name.startswith(LANG_PREFIX):
        return False
    base = name[: -len(".weight")]
    return base.rsplit(".", 1)[-1] in LINEAR_SUFFIXES


def quantize_group128(w: torch.Tensor, group_size: int):
    """w: [N, K] bf16 -> (weight_packed int32 [N, K/4], weight_scale bf16 [N, K/G])."""
    assert w.dim() == 2, f"expected 2D, got {w.shape}"
    N, K = w.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    assert K % 4 == 0, f"K={K} not divisible by 4 (packing)"

    wf = w.to(torch.float32)
    wg = wf.view(N, K // group_size, group_size)                # [N, G, gs]
    amax = wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)  # [N, G, 1]
    scale_f = amax / 127.0                                       # [N, G, 1] f32
    q = torch.round(wg / scale_f).clamp(-127, 127).to(torch.int8)  # [N, G, gs]
    q = q.view(N, K)                                            # [N, K]
    u8 = (q.to(torch.int16) + 128).to(torch.uint8)              # [N, K]
    packed = u8.view(torch.int32).contiguous()                  # [N, K/4]
    scale = scale_f.to(w.dtype).view(N, K // group_size).contiguous()  # [N, K/G]
    return packed, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()

    src, dst = args.src, args.dst
    os.makedirs(dst, exist_ok=True)

    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    # safetensors metadata must be str->str; stringify any non-str values.
    metadata = {
        k: (v if isinstance(v, str) else str(v))
        for k, v in (index.get("metadata") or {}).items()
    }

    # Copy non-safetensors files (config, tokenizer, etc.)
    for fn in os.listdir(src):
        p = os.path.join(src, fn)
        if os.path.isfile(p) and not fn.endswith(".safetensors") and fn != "model.safetensors.index.json":
            shutil.copy2(p, os.path.join(dst, fn))

    # Determine which tensors to quantize (must be 2D linears).
    to_quant = {k for k in weight_map if is_quant_target(k)}
    print(f"quantizing {len(to_quant)} linear tensors (group_size={args.group_size})")

    # Process each source safetensors file, write a corresponding output file.
    src_files = sorted(
        {weight_map[k] for k in weight_map},
        key=lambda s: int(re.search(r"model-(\d+)-of", s).group(1)) if re.search(r"model-(\d+)-of", s) else 0,
    )
    # Keep any extra safetensors (e.g. model-auxiliary) that are not in weight_map.
    all_src_st = sorted(
        fn for fn in os.listdir(src) if fn.endswith(".safetensors")
    )
    ordered = [f for f in all_src_st if f in src_files] + [f for f in all_src_st if f not in src_files]

    new_weight_map = {}
    total_size = 0
    n_packed = 0
    for fn in ordered:
        sp = os.path.join(src, fn)
        out_tensors = {}
        with safe_open(sp, framework="pt") as st:
            for k in st.keys():
                t = st.get_tensor(k)
                if k in to_quant:
                    packed, scale = quantize_group128(t, args.group_size)
                    out_tensors[k.replace(".weight", ".weight_packed")] = packed
                    out_tensors[k.replace(".weight", ".weight_scale")] = scale
                    n_packed += 1
                    del t
                else:
                    out_tensors[k] = t
        dp = os.path.join(dst, fn)
        save_file(out_tensors, dp, metadata=metadata or None)
        # Update weight map + sizes (from in-memory tensors, no reload).
        for k, t in out_tensors.items():
            new_weight_map[k] = fn
            total_size += t.numel() * t.element_size()
        print(f"  wrote {fn}: {len(out_tensors)} tensors")

    # Rebuild index.
    new_index = {"metadata": metadata, "weight_map": new_weight_map, "total_size": total_size}
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    # Add quantization_config to config.json (match lued / compressed-tensors format).
    cfg_path = os.path.join(dst, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = {
        "config_groups": {
            "group_0": {
                "format": "pack-quantized",
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": args.group_size,
                    "num_bits": 8,
                    "observer": "memoryless_minmax",
                    "observer_kwargs": {},
                    "scale_dtype": None,
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int",
                    "zp_dtype": None,
                },
            }
        },
        "format": "pack-quantized",
        "global_compression_ratio": None,
        # Ignore everything that is NOT a language-model linear layer.
        # IMPORTANT: the ignore list is matched against the CONSTRUCTION prefix
        # (used by get_quant_method at layer-build time), not the final vllm
        # weight name. For the qwen3_5 multimodal wrapper the language linears
        # are built under `language_model.model.layers.N.<proj>` (the wrapper
        # nests the LM under self.language_model -> self.model). Visual/MTP/
        # embed/norm/lm_head are not under that path, so they stay bf16.
        # (The `.*` makes it robust to any leading prefix.)
        # Second rule: the GDN `in_proj_ba` layer (merged in_proj_a+in_proj_b,
        # N=48) is never a multiple of 32 at any TP, so the int8 kernel cannot
        # implement it. Keep it bf16 (its source weights are left un-packed).
        # NOTE: for a fused layer, should_ignore_layer checks the UNFUSED shard
        # names (in_proj_b / in_proj_a) against the ignore list, NOT the fused
        # name. So the rule must match the shard names, hence `in_proj_[ab]$`.
        "ignore": [
            "re:^(?!.*language_model\\.model\\.layers\\.)",
            "re:.*in_proj_[ab]$",
        ],
        "quantization_type": "int",
        # Required: vllm uses quant_method to select the quant config class.
        # Without it, layers are created as plain bf16 (no weight_packed) and
        # loading fails with AttributeError on the merged linear layer.
        "quant_method": "compressed-tensors",
        "version": "0.1.0",
        "quantization_status": "compressed",
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nDONE: {n_packed} tensors quantized -> {dst}")
    print(f"total_size={total_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()
