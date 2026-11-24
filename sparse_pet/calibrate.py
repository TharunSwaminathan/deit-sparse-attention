"""Validation-only coarse-grid calibration for 2x2 study (no Optuna).

All four variants are evaluated against the SAME dense model checkpoint.
Select closest settings per requested realized token-level sparsity and save
frozen configs. This does not inspect or access official test data for tuning.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict

import torch

from .attention import patch_attention_layers, parse_layer_indices
from .common import build_datasets_for_model, evaluate_model, load_dense_checkpoint, save_json, sha256_file
from .evaluate import VARIANTS, average_wrapper_stats, make_config
from .selection import select_calibration_candidate, selection_warnings, calibration_grid_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--layers", required=True, help="0-based comma-separated blocks chosen from validation, e.g. 8,9,10")
    ap.add_argument("--data-root", default="data/downloads")
    ap.add_argument("--manifest", default="outputs/split_manifest.json")
    ap.add_argument("--output", default="outputs/calibration.json")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--images", type=int, default=64, help="fixed validation calibration subset")
    ap.add_argument("--targets", nargs="+", type=float, default=[.25, .50])
    ap.add_argument("--top-k-grid", nargs="+", type=int, default=[12, 24, 36],
                    help="Try each k for hybrid C/D; the Top-p A/B k is irrelevant")
    ap.add_argument("--p-grid", nargs="+", type=float,
                    default=[.60, .75, .85, .95, .99])
    ap.add_argument("--similarity-grid", nargs="+", type=float, default=[.60, .80])
    ap.add_argument("--tolerance", type=float, default=.03)
    ap.add_argument("--min-activity", type=float, default=.002,
                    help="minimum coarse block-pair addition required for each claimed mechanism")
    ap.add_argument("--show-grid-size", action="store_true",
                    help="display candidate count and exit without loading a checkpoint or dataset")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if not args.top_k_grid or any(k < 1 or k > 49 for k in args.top_k_grid):
        ap.error("--top-k-grid values must be within 1..49")
    if not args.p_grid or any(not 0 < p <= 1 for p in args.p_grid):
        ap.error("--p-grid values must be within (0, 1]")
    if not args.similarity_grid or any(not -1 <= t <= 1 for t in args.similarity_grid):
        ap.error("--similarity-grid must contain values within [-1, 1]")
    if not args.targets or any(not 0 <= t <= 1 for t in args.targets):
        ap.error("--targets must be within [0, 1]")
    if not 0 <= args.tolerance < 1 or not 0 <= args.min_activity <= 1:
        ap.error("--tolerance and --min-activity must be within [0, 1)")
    print(f"Candidate configurations: {calibration_grid_size(args.p_grid, args.top_k_grid, args.similarity_grid)}", flush=True)
    if args.show_grid_size:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_dense_checkpoint(args.checkpoint, device)
    _, val_loader, _, _ = build_datasets_for_model(
        model, args.data_root, args.manifest, args.batch_size,
        args.workers, args.download)
    layer_indices = parse_layer_indices(args.layers, len(model.blocks))
    wrappers = patch_attention_layers(model, mask_config=None, layer_indices=layer_indices, diagnostics=True)
    candidates = {}
    for variant, (method, protected) in VARIANTS.items():
        candidates[variant] = []
        similarities = args.similarity_grid if protected else [None]
        ks = args.top_k_grid if method == "hybrid" else [4]
        for p, threshold, top_k in itertools.product(args.p_grid, similarities, ks):
            config = make_config(variant, p, top_k, threshold)
            for w in wrappers:
                w.mask_config = config
                w.reset_diagnostics()
            with torch.inference_mode():
                scores = evaluate_model(model, val_loader, device, max_images=args.images)
            stats = average_wrapper_stats(wrappers)
            row = {"variant": variant, "p": p, "top_k": top_k,
                   "similarity_threshold": threshold, "realized_sparsity": stats["realized_sparsity"],
                   "calibration_accuracy": scores["accuracy"],
                   "retained_mass": stats["retained_mass"],
                   "topk_added_fraction": stats["topk_added_fraction"],
                   "protection_added_fraction": stats["protection_added_fraction"],
                   "protected_query_block_fraction": stats["protected_query_block_fraction"],
                   "protected_key_block_fraction": stats["protected_key_block_fraction"]}
            candidates[variant].append(row)
            print(f'{variant}: p={p:.2f}, k={top_k}, sim={threshold}, sparsity={row["realized_sparsity"]:.3f}, k-added={row["topk_added_fraction"]:.3f}, protection-added={row["protection_added_fraction"]:.3f}', flush=True)
    selected = {}
    warnings = []
    for variant in VARIANTS:
        for target in args.targets:
            key = f"{variant}_{round(target * 100)}"
            chosen = select_calibration_candidate(
                candidates[variant], variant, target,
                tolerance=args.tolerance, min_activity=args.min_activity)
            selected[key] = chosen
            warnings.extend(selection_warnings(key, chosen, args.min_activity))
    payload = {"checkpoint": args.checkpoint, "checkpoint_sha256": sha256_file(args.checkpoint),
               "split": "validation_only",
               "layer_indices": layer_indices,
               "calibration_images": args.images, "tolerance": args.tolerance,
               "selection_policy": "budget feasibility first, then nonzero mechanism effect, then closeness to target",
               "min_activity": args.min_activity,
               "selected": selected, "candidate_results": candidates,
               "calibration_warnings": warnings,
               "candidate_grid": {"p": args.p_grid, "top_k": args.top_k_grid, "similarity": args.similarity_grid}}
    save_json(args.output, payload)
    print(json.dumps({"selected": selected, "warnings": warnings}, indent=2))
    print(f"FROZEN CALIBRATION SAVED: {args.output}. Review infeasible targets before test.")

if __name__ == "__main__":
    main()
