"""No-dataset DeiT integration gate for Colab.

Uses random weights by default (no download). An optional real fine-tuned
checkpoint checks the exact weights used for the final study.
"""
from __future__ import annotations

import argparse
import copy
import json

import torch

from .attention import patch_attention_layers, parse_layer_indices
from .common import load_dense_checkpoint, load_model
from .masking import MaskConfig


def _delta(reference: torch.Tensor, output: torch.Tensor) -> dict:
    difference = (reference.float() - output.float()).abs()
    return {"max_absolute_difference": float(difference.max().item()),
            "mean_absolute_difference": float(difference.mean().item())}


def run_smoke(model, indices, device):
    """Compare native model, manual-dense wrapper, all-kept, and sparse mask."""
    model.eval().to(device)
    sample = torch.randn(1, 3, 224, 224, device=device)
    with torch.inference_mode():
        native = model(sample)
        unmasked = copy.deepcopy(model)
        patch_attention_layers(unmasked, None, indices)
        unmasked_logits = unmasked(sample)
        all_kept = copy.deepcopy(model)
        patch_attention_layers(all_kept, MaskConfig(p=1.0), indices)
        all_kept_logits = all_kept(sample)
        sparse = copy.deepcopy(model)
        wrappers = patch_attention_layers(
            sparse, MaskConfig(method="hybrid", p=0.05, top_k=1), indices,
            diagnostics=True)
        sparse_logits = sparse(sample)
    dense_delta = _delta(native, unmasked_logits)
    full_mask_delta = _delta(native, all_kept_logits)
    sparse_delta = _delta(native, sparse_logits)
    if not torch.allclose(native, unmasked_logits, rtol=1e-4, atol=1e-4):
        raise AssertionError("Unmasked wrapper disagrees with native timm attention")
    if not torch.allclose(native, all_kept_logits, rtol=1e-4, atol=1e-4):
        raise AssertionError("All-retained mask disagrees with native timm attention")
    if not torch.isfinite(sparse_logits).all():
        raise AssertionError("Sparse inference produced NaN/Inf")
    return {"layers": indices, "native_vs_unmasked": dense_delta,
            "native_vs_all_kept": full_mask_delta,
            "native_vs_sparse": sparse_delta,
            "masked_layer_sparsity": [w.stats().get("realized_sparsity") for w in wrappers],
            "pass_equivalence": True,
            "note": "Sparse logits need not change when only the final block is masked and CLS rows are kept."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="optional 37-way dense checkpoint; otherwise no pretrained weights")
    ap.add_argument("--layers", default="8,9,10", help="0-based DeiT attention blocks")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--output", help="optional JSON output file")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.checkpoint:
        model, _ = load_dense_checkpoint(args.checkpoint, device)
    else:
        model = load_model(pretrained=False, num_classes=37).to(device)
    indices = parse_layer_indices(args.layers, len(model.blocks))
    result = run_smoke(model, indices, device)
    result.update({"device": args.device, "checkpoint": args.checkpoint or "random_weights"})
    if args.output:
        from .common import save_json
        save_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
