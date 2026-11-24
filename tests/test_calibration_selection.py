import pytest
import torch

from sparse_pet.compare_masks import binary_mask_comparison
from sparse_pet.selection import (
    calibration_grid_size, mechanism_activity,
    select_calibration_candidate, selection_warnings,
)


def row(sparsity, topk=0., protection=0., accuracy=.9, p=.8):
    return {"p": p, "realized_sparsity": sparsity,
            "topk_added_fraction": topk, "protection_added_fraction": protection,
            "calibration_accuracy": accuracy}


def test_grid_size_counts_no_protection_and_protected_grid():
    assert calibration_grid_size([.7, .8], [4, 20], [.5, .8]) == 2 * (1 + 2 + 2 + 4)


def test_pick_active_hybrid_when_in_tolerance():
    result = select_calibration_candidate(
        [row(.50, topk=0.), row(.515, topk=.01)], "C", .5, tolerance=.03)
    assert result["feasible_within_tolerance"]
    assert result["mechanisms_active"]
    assert result["realized_sparsity"] == .515


def test_pick_both_mechanisms_in_D():
    result = select_calibration_candidate([
        row(.5, topk=.1), row(.51, topk=.1, protection=.02),
        row(.50, protection=.1)], "D", .5)
    assert result["mechanisms_active"]
    assert result["realized_sparsity"] == .51
    assert result["valid_matched_activity_comparison"]


def test_infeasible_never_claims_matched_even_if_active():
    result = select_calibration_candidate([row(.6, topk=.1)], "C", .5, tolerance=.03)
    assert not result["feasible_within_tolerance"]
    assert not result["valid_matched_activity_comparison"]
    assert any("NOT a matched-budget" in w for w in selection_warnings("C_50", result, .002))


def test_inactive_protection_is_reported_not_silent():
    result = select_calibration_candidate([row(.501)], "B", .50)
    assert result["feasible_within_tolerance"]
    assert not result["mechanisms_active"]
    assert len(selection_warnings("B_50", result, .002)) >= 2


def test_activity_flags_by_variant():
    sample = row(.5, topk=.01, protection=0)
    assert mechanism_activity(sample, "A", .002) == {}
    assert mechanism_activity(sample, "C", .002)["top_k_active"]
    assert not mechanism_activity(sample, "D", .002)["protection_active"]
    with pytest.raises(ValueError):
        mechanism_activity(sample, "X", .002)


def test_reject_empty_and_nonfinite_calibration():
    with pytest.raises(ValueError):
        select_calibration_candidate([], "A", .5)
    with pytest.raises(ValueError):
        select_calibration_candidate([row(float("nan"))], "A", .5)


def test_mask_overlap_matches_exact_and_different_masks():
    a = torch.ones(2, 4, 4, dtype=torch.bool)
    b = a.clone()
    same = binary_mask_comparison(a, b)
    assert same["changed_edges"] == 0 and same["union"] == a.numel()
    b[:, 2, 3] = False
    different = binary_mask_comparison(a, b)
    assert different["changed_edges"] == 2
    assert different["changed_patch_edges"] == 2
    with pytest.raises(ValueError):
        binary_mask_comparison(a, b.float())
