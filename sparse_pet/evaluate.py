"""Evaluate dense or sparse attention, without changing model parameters.

Validation: manually select --variant, --p, etc.
Official test: require --frozen-config from validation-only calibration and --label.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .attention import patch_attention_layers, parse_layer_indices
from .common import build_datasets_for_model, evaluate_model, load_dense_checkpoint, save_json, sha256_file
from .masking import MaskConfig

VARIANTS = {"A": ("top_p", False), "B": ("top_p", True),
            "C": ("hybrid", False), "D": ("hybrid", True)}


def make_config(variant, p, top_k, similarity_threshold):
    if variant == "dense" or variant == "unmasked":
        return None
    method, protected = VARIANTS[variant]
    if protected and similarity_threshold is None:
        raise ValueError("protected variant needs similarity_threshold")
    return MaskConfig(method=method, p=p, top_k=top_k,
                      similarity_threshold=similarity_threshold if protected else None)


def benchmark(model, device, repetitions=20, warmup=5, batch_size=1):
    """End-to-end forward latency. For timing, disable diagnostic GPU syncs."""
    model.eval()
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    times = []
    with torch.inference_mode():
        for i in range(warmup + repetitions):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration_ms = (time.perf_counter() - start) * 1000
            if i >= warmup:
                times.append(duration_ms)
    return {"median_ms": statistics.median(times), "repetitions": repetitions,
            "timing_batch_size": batch_size, "precision": "fp32"}


def average_wrapper_stats(wrappers):
    rows = [w.stats() for w in wrappers if w.stats()]
    if not rows:
        return {}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="data/downloads")
    ap.add_argument("--manifest", default="outputs/split_manifest.json")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--variant", choices=["dense", "unmasked", *VARIANTS], default="A")
    ap.add_argument("--layers", help="0-based attention blocks, e.g. 8,9,10; taken from calibration when frozen config supplied")
    ap.add_argument("--p", type=float, default=.7)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--similarity-threshold", type=float, default=None)
    ap.add_argument("--frozen-config", help="validation-only calibration JSON; required for test")
    ap.add_argument("--label", help="e.g. A_25, B_50; used with --frozen-config")
    ap.add_argument("--max-images", type=int, default=None, help="validation pilot only")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--output", default="outputs/evaluation.json")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.split == "test":
        if not args.frozen_config:
            raise ValueError("Test is locked: first calibrate on validation and provide --frozen-config")
        if args.max_images is not None:
            raise ValueError("Official test evaluation must use the entire test set")
    frozen_layers = None
    if args.frozen_config:
        payload = json.loads(Path(args.frozen_config).read_text())
        if payload.get("checkpoint_sha256") is not None:
            if sha256_file(args.checkpoint) != payload["checkpoint_sha256"]:
                raise ValueError("Checkpoint differs from the one used for frozen calibration")
        frozen_layers = payload.get("layer_indices")
        if args.layers is not None and frozen_layers is not None:
            supplied = sorted(int(i.strip()) for i in args.layers.split(","))
            if supplied != frozen_layers:
                raise ValueError("Layer choices differ from frozen validation calibration")
        if args.label:
            if args.label not in payload["selected"]:
                raise KeyError(f"unknown calibration label {args.label}")
            c = payload["selected"][args.label]
            args.variant = c["variant"]
            args.p = c["p"]
            args.top_k = c["top_k"]
            args.similarity_threshold = c["similarity_threshold"]
        elif args.split == "test" and args.variant not in ("dense", "unmasked"):
            raise ValueError("Specify --label for calibrated sparse test evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_dense_checkpoint(args.checkpoint, device)
    _, val_loader, test_loader, _ = build_datasets_for_model(
        model, args.data_root, args.manifest, args.batch_size, args.workers,
        download=args.download)
    mask_config = make_config(args.variant, args.p, args.top_k, args.similarity_threshold)
    wrappers = []
    layer_indices = None
    if args.variant != "dense":
        if args.layers is None and frozen_layers is None:
            raise ValueError("Provide --layers (selected on validation), or a frozen calibration")
        layer_indices = (parse_layer_indices(args.layers, len(model.blocks)) if args.layers is not None
                         else parse_layer_indices(",".join(map(str, frozen_layers)), len(model.blocks)))
        wrappers = patch_attention_layers(model, mask_config, layer_indices=layer_indices, diagnostics=True)
    loader = val_loader if args.split == "val" else test_loader
    scores = evaluate_model(model, loader, device, max_images=args.max_images)
    diagnostics = average_wrapper_stats(wrappers)
    for w in wrappers:
        w.collect_diagnostics = False
    result = {"split": args.split, "variant": args.variant,
              "layer_indices": layer_indices,
              "mask_config": asdict(mask_config) if mask_config else None,
              "scores": scores, "diagnostics": diagnostics,
              "checkpoint": args.checkpoint, "device": str(device)}
    if args.benchmark:
        result["latency"] = benchmark(model, device, repetitions=20, warmup=5)
    save_json(args.output, result)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
