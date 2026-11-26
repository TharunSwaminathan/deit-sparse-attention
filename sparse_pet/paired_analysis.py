"""Paired accuracy comparisons for existing prediction exports; no model inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CONFIGURATIONS = ("dense", "unmasked", *(f"{v}_{t}" for t in (25, 50) for v in "ABCD"))
COLUMNS = {"split", "configuration", "image_id", "image_index", "ground_truth", "predicted_class", "correct"}


def read_run(directory, configuration, split):
    """Read matching CSV/JSON; reject inconsistent labels, IDs and accuracy."""
    directory = Path(directory)
    csv_path, json_path = directory / f"{configuration}.csv", directory / f"{configuration}.json"
    rows = {}
    prefix = {"val": "trainval", "test": "test"}[split]
    limit = {"val": 3680, "test": 3669}[split]
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(COLUMNS) or set(reader.fieldnames) != COLUMNS:
            raise ValueError(f"{csv_path}: unexpected CSV columns")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{csv_path}: malformed row")
            if row["split"] != split or row["configuration"] != configuration:
                raise ValueError(f"{csv_path}: wrong split or configuration")
            index, truth, pred, correct = (int(row[k]) for k in ("image_index", "ground_truth", "predicted_class", "correct"))
            image_id = row["image_id"]
            if not 0 <= index < limit or image_id != f"{prefix}:{index}":
                raise ValueError(f"{csv_path}: invalid image ID/index")
            if not (0 <= truth < 37 and 0 <= pred < 37) or correct != int(truth == pred):
                raise ValueError(f"{csv_path}: invalid class or correctness indicator")
            if image_id in rows:
                raise ValueError(f"{csv_path}: duplicate image ID {image_id}")
            rows[image_id] = (truth, pred, correct)
    if not rows:
        raise ValueError(f"{csv_path}: no predictions")
    if split == "test" and set(rows) != {f"test:{i}" for i in range(3669)}:
        raise ValueError(f"{csv_path}: official test must contain all 3669 IDs")
    payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    export, scores = payload["prediction_export"], payload["scores"]
    expected_variant = configuration.split("_")[0]
    if (payload["split"] != split or payload["variant"] != expected_variant or
            export["configuration"] != configuration or export["id_scheme"] != "official_split:zero_based_index" or
            export["n"] != len(rows) or scores["n"] != len(rows)):
        raise ValueError(f"{json_path}: metadata or count mismatch")
    accuracy = sum(r[2] for r in rows.values()) / len(rows)
    if not math.isclose(accuracy, float(scores["accuracy"]), rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"{json_path}: CSV and JSON accuracy mismatch")
    fingerprints = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (csv_path, json_path)}
    return rows, fingerprints


def paired_comparison(reference, candidate, repetitions=10000, seed=42):
    """Percentile bootstrap of candidate-minus-reference accuracy (percentage points).

    Multinomial draws of paired correctness differences (-1, 0, +1) are exactly
    equivalent in distribution to resampling image pairs with replacement.
    """
    if repetitions < 1000 or seed < 0:
        raise ValueError("Use at least 1000 bootstrap repetitions and a nonnegative seed")
    if not reference or set(reference) != set(candidate):
        raise ValueError("Prediction ID sets differ or are empty")
    ids = sorted(reference)
    a = np.asarray([reference[i] for i in ids], dtype=np.int64)
    b = np.asarray([candidate[i] for i in ids], dtype=np.int64)
    if not np.array_equal(a[:, 0], b[:, 0]):
        raise ValueError("Ground-truth labels differ for matching image IDs")
    n = len(ids)
    diff = b[:, 2] - a[:, 2]
    wins, losses = int(np.sum(diff == 1)), int(np.sum(diff == -1))
    ties = n - wins - losses
    counts = np.random.default_rng(seed).multinomial(n, [losses/n, ties/n, wins/n], size=repetitions)
    samples = 100 * (counts[:, 2] - counts[:, 0]) / n
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"n": n, "dense_accuracy": float(a[:, 2].mean()), "candidate_accuracy": float(b[:, 2].mean()),
            "accuracy_delta_pp": 100 * (wins-losses)/n,
            "ci95_delta_pp": [float(lo), float(hi)],
            "candidate_only_correct": wins, "dense_only_correct": losses,
            "both_correct": int(np.sum((a[:, 2] == 1) & (b[:, 2] == 1))),
            "both_wrong": int(np.sum((a[:, 2] == 0) & (b[:, 2] == 0))),
            "prediction_disagreements": int(np.sum(a[:, 1] != b[:, 1])),
            "prediction_disagreement_rate": float(np.mean(a[:, 1] != b[:, 1])),
            "degenerate_bootstrap": bool(np.all(diff == diff[0]))}


def analyze(directory, split, configurations=CONFIGURATIONS, manifest=None, repetitions=10000, seed=42):
    configurations = tuple(configurations)
    if (len(configurations) < 2 or "dense" not in configurations or len(set(configurations)) != len(configurations)
            or not set(configurations).issubset(CONFIGURATIONS)):
        raise ValueError("Select dense and at least one other unique known configuration")
    if split not in ("val", "test"):
        raise ValueError("Unknown split")
    if split == "test" and set(configurations) != set(CONFIGURATIONS):
        raise ValueError("Official test analysis requires all ten configurations")
    reference, fingerprints = read_run(directory, "dense", split)
    if split == "val":
        if manifest is None:
            raise ValueError("Validation analysis requires the original split manifest")
        path = Path(manifest)
        allowed = {f"trainval:{i}" for i in json.loads(path.read_text(encoding="utf-8-sig"))["val_indices"]}
        if not set(reference).issubset(allowed):
            raise ValueError("Validation IDs outside the saved validation manifest")
        fingerprints["validation_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    results = {}
    for configuration in configurations:
        if configuration == "dense":
            continue
        candidate, hashes = read_run(directory, configuration, split)
        fingerprints.update(hashes)
        results[configuration] = paired_comparison(reference, candidate, repetitions, seed)
    return {"split": split, "reference": "dense", "n": len(reference), "configurations": list(configurations),
            "bootstrap": {"method": "paired image-level percentile, IID resampling", "repetitions": repetitions,
                          "seed": seed, "confidence_level": 0.95, "multiplicity_adjusted": False,
                          "numpy_version": np.__version__},
            "input_sha256": fingerprints, "comparisons": results,
            "limitations": ["Intervals are marginal, exploratory, and not adjusted for multiple comparisons.",
                            "One checkpoint; no training-seed variability measured. Degenerate intervals do not prove equivalence.",
                            "CSV/JSON consistency does not verify checkpoint, dataset release, or approved settings; verify run provenance separately.",
                            "Validation pilots are functionality checks, not final inference."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/split_manifest.json"))
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sources = {(args.input_dir / f"{c}.{ext}").resolve() for c in args.configurations for ext in ("csv", "json")}
    if args.output.resolve() in sources | {args.manifest.resolve()}:
        parser.error("Output cannot overwrite an input file")
    report = analyze(args.input_dir, args.split, args.configurations, args.manifest, args.repetitions, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"PAIRED ANALYSIS COMPLETE: {report['n']} {args.split} images")
    for name, row in report["comparisons"].items():
        lo, hi = row["ci95_delta_pp"]
        print(f"{name}: delta {row['accuracy_delta_pp']:+.3f} pp; 95% CI [{lo:+.3f}, {hi:+.3f}] pp; "
              f"wins/losses {row['candidate_only_correct']}/{row['dense_only_correct']}; "
              f"prediction disagreements {row['prediction_disagreements']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
