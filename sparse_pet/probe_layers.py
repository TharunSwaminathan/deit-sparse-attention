"""Validation-only preliminary attention-layer probe.

Compares single-layer mask insertion to the fixed dense checkpoint on the
same validation images. Select layers *before* calibrating A-D; this is not a
method for tuning on the official test set.
"""
from __future__ import annotations

import argparse
import copy
import json
import torch

from .attention import patch_attention_layers, parse_layer_indices
from .common import build_datasets_for_model, load_dense_checkpoint, save_json
from .masking import MaskConfig


@torch.inference_mode()
def gather_images(loader, device, limit):
    inputs = []
    for images, _ in loader:
        remain = limit - sum(x.size(0) for x in inputs)
        if remain <= 0:
            break
        inputs.append(images[:remain].to(device))
    if not inputs:
        raise ValueError("No validation images")
    return inputs


@torch.inference_mode()
def compute_logits(model, batches):
    model.eval()
    return torch.cat([model(images).float() for images in batches], dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--layers", default="3,6,8,9,10,11")
    ap.add_argument("--data-root", default="data/downloads")
    ap.add_argument("--manifest", default="outputs/split_manifest.json")
    ap.add_argument("--images", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--p", type=float, default=0.1, help="aggressive pilot mask to reveal sensitivity")
    ap.add_argument("--output", default="outputs/layer_probe.json")
    args = ap.parse_args()
    if args.images < 1:
        raise ValueError("--images must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_dense_checkpoint(args.checkpoint, device)
    indices = parse_layer_indices(args.layers, len(model.blocks))
    _, validation_loader, _, _ = build_datasets_for_model(
        model, args.data_root, args.manifest, args.batch_size, args.workers)
    images = gather_images(validation_loader, device, args.images)
    dense_logits = compute_logits(model, images)
    reports = []
    for index in indices:
        # One at a time: a clean attention module in every other layer.
        experiment = copy.deepcopy(model)
        wrapper, = patch_attention_layers(
            experiment, MaskConfig(method="top_p", p=args.p), [index], diagnostics=True)
        sparse_logits = compute_logits(experiment, images)
        reports.append({
            "layer": index,
            "mean_abs_logit_change": float((dense_logits - sparse_logits).abs().mean().item()),
            "max_abs_logit_change": float((dense_logits - sparse_logits).abs().max().item()),
            "prediction_flip_fraction": float((dense_logits.argmax(-1) != sparse_logits.argmax(-1)).float().mean().item()),
            "attention": wrapper.stats(),
        })
        del experiment
    output = {"split": "validation_only", "checkpoint": args.checkpoint,
              "n_images": dense_logits.shape[0], "p": args.p,
              "candidate_layers": indices, "results": reports,
              "warning": "Do not pick layers using the test split. The final block alone can have zero impact on CLS predictions."}
    save_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
