"""Pure validation-only calibration selection rules; independently unit-testable.

A sparse variant is scientifically informative only if its added masking rule
actually changes some coarse block pairs. Preserve the hard target-sparsity
tolerance before preferring mechanism activity: never mislabel unmatched runs.
"""
from __future__ import annotations

import math


def calibration_grid_size(p_grid, top_k_grid, similarity_grid):
    """Total candidate forward evaluations across four proposal variants."""
    return len(p_grid) * (1 + len(similarity_grid) + len(top_k_grid)
                          + len(top_k_grid) * len(similarity_grid))


def mechanism_activity(row, variant: str, min_activity: float) -> dict:
    """Whether the mechanism-specific additions are substantively nonzero.

    Coarse-pair fraction is computed before expanding blocks to patch tokens.
    It measures *within-candidate* additions, not cross-variant differences.
    Cross-variant differences require compare_masks.py.
    """
    if variant not in "ABCD" or len(variant) != 1:
        raise ValueError(f"Unknown sparse variant: {variant!r}")
    activity = {}
    if variant in ("C", "D"):
        activity["top_k_active"] = float(row["topk_added_fraction"]) >= min_activity
    if variant in ("B", "D"):
        activity["protection_active"] = float(row["protection_added_fraction"]) >= min_activity
    return activity


def select_calibration_candidate(rows, variant: str, target: float,
                                 tolerance: float = .03,
                                 min_activity: float = .002):
    """First match budget, then prefer active mechanisms, then match closely.

    If no candidate is within tolerance, return the closest observed one but
    mark it infeasible; users must not report that point as budget-matched.
    """
    if not rows:
        raise ValueError("Cannot calibrate from an empty list")
    if not 0 <= target <= 1 or not 0 <= tolerance <= 1 or not 0 <= min_activity <= 1:
        raise ValueError("Invalid target, tolerance, or activity threshold")
    for row in rows:
        if (not math.isfinite(float(row["realized_sparsity"])) or
                not 0 <= float(row["realized_sparsity"]) <= 1):
            raise ValueError("Non-finite or invalid measured sparsity")
    feasible = [r for r in rows if abs(r["realized_sparsity"] - target) <= tolerance]
    pool = feasible if feasible else rows
    active = [r for r in pool if all(mechanism_activity(r, variant, min_activity).values())]
    used_active = bool(active)
    if active:
        pool = active
    chosen = min(pool, key=lambda r: (abs(r["realized_sparsity"] - target),
                                       -float(r.get("calibration_accuracy", 0.0)),
                                       float(r["p"])))
    status = mechanism_activity(chosen, variant, min_activity)
    return {
        **chosen, "target_sparsity": target,
        "absolute_error": abs(chosen["realized_sparsity"] - target),
        "feasible_within_tolerance": bool(feasible),
        "mechanism_activity": status,
        "mechanisms_active": all(status.values()),
        "active_candidate_available_within_tolerance": (
            used_active and bool(feasible)),
        "valid_matched_activity_comparison": bool(feasible) and all(status.values()),
    }


def selection_warnings(label: str, selection: dict, min_activity: float):
    warnings = []
    if not selection["feasible_within_tolerance"]:
        warnings.append(
            f"{label}: sparsity missed target by {selection['absolute_error']:.3f}; "
            "NOT a matched-budget result")
    for flag, active in selection["mechanism_activity"].items():
        if not active:
            warnings.append(
                f"{label}: {flag} is false (<{min_activity:.3%} of coarse pairs "
                "added). The corresponding ablation is inactive")
    if selection["feasible_within_tolerance"] and not selection["mechanisms_active"]:
        warnings.append(
            f"{label}: no configuration at this target satisfied all activity "
            "requirements; report limitation or revise validation search")
    return warnings
