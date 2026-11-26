"""Compare frozen A/B/C/D masks on identical *dense* layer inputs.

Validation-only mechanism diagnostic. It records Q/K in native dense DeiT
without modifying activations, then builds each candidate mask from exactly
the same Q/K tensors. This distinguishes true cross-method mask differences
from within-method diagnostics such as topk_added_fraction.

IMPORTANT: This is NOT a prediction evaluation or actual masked-model
sparsity measurement. Later layers in a multi-layer sparse model may receive
different features. Full evaluation must be run separately on validation.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import torch

from .common import build_datasets_for_model, load_dense_checkpoint, save_json, sha256_file
from .evaluate import make_config
from .masking import build_attention_mask


def binary_mask_comparison(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compare masks of equal shape; 197x197 includes mandatory CLS edges."""
    if a.shape != b.shape or a.dtype != torch.bool or b.dtype != torch.bool:
        raise ValueError("Masks must have equal shapes and boolean dtype")
    if a.ndim < 2:
        raise ValueError("Expected query and key axes")
    change = (a ^ b)
    a_patch = a[..., 1:, 1:]
    b_patch = b[..., 1:, 1:]
    return {
        "changed_edges": int(change.sum().item()),
        "total_edges": a.numel(),
        "changed_patch_edges": int((a_patch ^ b_patch).sum().item()),
        "total_patch_edges": a_patch.numel(),
        "intersection": int((a & b).sum().item()),
        "union": int((a | b).sum().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--frozen-config", required=True)
    ap.add_argument("--data-root", default="data/downloads")
    ap.add_argument("--manifest", default="outputs/split_manifest.json")
    ap.add_argument("--images", type=int, default=64,
                    help="fixed first N validation images for diagnostic")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--targets", nargs="+", type=int, default=[25, 50])
    ap.add_argument("--output", default="outputs/mask_overlap_val.json")
    args = ap.parse_args()
    if args.images < 1 or args.batch_size < 1 or not args.targets:
        ap.error("--images, --batch-size and --targets must be positive")

    frozen = json.loads(Path(args.frozen_config).read_text())
    if sha256_file(args.checkpoint) != frozen["checkpoint_sha256"]:
        raise ValueError("Frozen calibration belongs to a different checkpoint")
    layers = frozen["layer_indices"]
    labels = [f"{variant}_{target}" for target in args.targets for variant in "ABCD"]
    missing = [label for label in labels if label not in frozen["selected"]]
    if missing:
        raise ValueError(f"Missing frozen configurations: {missing}")

    configs = {}
    for label in labels:
        r = frozen["selected"][label]
        configs[label] = make_config(r["variant"], r["p"], r["top_k"],
                                     r["similarity_threshold"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_dense_checkpoint(args.checkpoint, device)
    model.eval()
    _, val_loader, _, _ = build_datasets_for_model(
        model, args.data_root, args.manifest, args.batch_size,
        args.workers, args.download)

    capture: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []
    for idx in layers:
        base = model.blocks[idx].attn
        def capture_qk(module, inputs, output, index=idx, attention=base):
            b, n, triple_dim = output.shape
            h = attention.num_heads
            if triple_dim % (3 * h):
                raise RuntimeError("Unexpected qkv shape for DeiT attention")
            head_dim = triple_dim // (3 * h)
            q, k, _ = output.reshape(b, n, 3, h, head_dim).permute(2, 0, 3, 1, 4).unbind(0)
            if hasattr(attention, "q_norm"):
                q = attention.q_norm(q)
            if hasattr(attention, "k_norm"):
                k = attention.k_norm(k)
            capture[index] = (q, k)

        handles.append(base.qkv.register_forward_hook(capture_qk))

    totals = {str(t): {"methods": {v: {"total_edges": 0, "masked_edges": 0,
                                            "coarse_topk_added_sum": 0.,
                                            "coarse_protection_added_sum": 0.,
                                            "n_samples": 0} for v in "ABCD"},
                       "pairs": {f"{a}-{b}": {"changed_edges": 0, "total_edges": 0,
                                                "changed_patch_edges": 0,
                                                "total_patch_edges": 0,
                                                "intersection": 0, "union": 0}
                                 for a, b in combinations("ABCD", 2)}} for t in args.targets}
    count = 0
    try:
        with torch.inference_mode():
            for x, _ in val_loader:
                if count >= args.images:
                    break
                x = x[: args.images - count].to(device, non_blocking=True)
                capture.clear()
                model(x)
                if set(capture) != set(layers):
                    raise RuntimeError("Failed to capture selected dense attention layers")
                count += len(x)
                for idx in layers:
                    q, k = capture[idx]
                    for target in args.targets:
                        masks = {}
                        target_stats = totals[str(target)]
                        for variant in "ABCD":
                            m, details = build_attention_mask(q, k, configs[f"{variant}_{target}"],
                                                             diagnostics=True)
                            masks[variant] = m
                            stats = target_stats["methods"][variant]
                            stats["total_edges"] += m.numel()
                            stats["masked_edges"] += (~m).sum().item()
                            stats["coarse_topk_added_sum"] += details["topk_added"].sum().item()
                            stats["coarse_protection_added_sum"] += details["protection_added"].sum().item()
                            stats["n_samples"] += details["coarse_keep"].numel()
                        for a, b in combinations("ABCD", 2):
                            info = binary_mask_comparison(masks[a], masks[b])
                            row = target_stats["pairs"][f"{a}-{b}"]
                            for key, value in info.items():
                                row[key] += value
    finally:
        for handle in handles:
            handle.remove()

    report = {"split": "validation_only", "checkpoint_sha256": sha256_file(args.checkpoint),
              "layers": layers, "images": count, "note":
              "All A/B/C/D masks were constructed from identical DENSE attention inputs. "
              "Actual multi-layer masked-model sparsity must be evaluated separately.",
              "targets": {}}
    for target in args.targets:
        data = totals[str(target)]
        for variant, row in data["methods"].items():
            row["measured_sparsity_on_dense_inputs"] = row["masked_edges"] / row["total_edges"]
            row["coarse_topk_added_fraction"] = row["coarse_topk_added_sum"] / row["n_samples"]
            row["coarse_protection_added_fraction"] = row["coarse_protection_added_sum"] / row["n_samples"]
        for pair, row in data["pairs"].items():
            row["changed_edge_fraction"] = row["changed_edges"] / row["total_edges"]
            row["changed_patch_edge_fraction"] = row["changed_patch_edges"] / row["total_patch_edges"]
            row["jaccard_kept_edges"] = (row["intersection"] / row["union"] if row["union"] else 1.)
        report["targets"][str(target)] = data
        print(f"Target {target}%: comparison of masks on identical dense Q/K:")
        for pair in ("A-B", "A-C", "B-D", "C-D"):
            row = data["pairs"][pair]
            print(f"  {pair}: patch-edge disagreement="
                  f"{row['changed_patch_edge_fraction']:.3%}; "
                  f"kept-mask Jaccard={row['jaccard_kept_edges']:.4f}")
    save_json(args.output, report)
    print(f"Saved validation-only comparison: {args.output}")


if __name__ == "__main__":
    main()
