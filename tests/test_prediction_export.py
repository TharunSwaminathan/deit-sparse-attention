import csv
import json
import sys

import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from sparse_pet.common import evaluate_model
from sparse_pet.predictions import evaluation_indices, write_predictions
from sparse_pet import evaluate


def fixture_dataset():
    logits = torch.zeros(6, 37)
    logits[range(6), [0, 2, 2, 3, 1, 5]] = 5
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    return TensorDataset(logits, labels)


def test_export_preserves_metrics_and_subset_ids(tmp_path):
    loader = DataLoader(Subset(fixture_dataset(), [4, 0, 3, 1]), batch_size=3)
    model = torch.nn.Identity()
    baseline = evaluate_model(model, loader, torch.device("cpu"))
    rows = []
    scores = evaluate_model(model, loader, torch.device("cpu"), prediction_rows=rows)
    assert scores == baseline
    assert rows == [(4, 4, 1), (0, 0, 0), (3, 3, 3), (1, 1, 2)]
    path = tmp_path / "val.csv"
    write_predictions(path, rows, "val", "A_25")
    with path.open(newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert [r["image_id"] for r in exported] == ["trainval:4", "trainval:0", "trainval:3", "trainval:1"]
    assert all(r["configuration"] == "A_25" and r["split"] == "val" for r in exported)
    assert sum(int(r["correct"]) for r in exported) / len(exported) == scores["accuracy"]


def test_partial_batch_and_nested_subset():
    subset = Subset(Subset(fixture_dataset(), [5, 3, 1, 0]), [2, 0, 1])
    loader = DataLoader(subset, batch_size=2)
    rows = []
    result = evaluate_model(torch.nn.Identity(), loader, torch.device("cpu"), max_images=1, prediction_rows=rows)
    assert result["n"] == 1
    assert rows == [(1, 1, 2)]


@pytest.mark.parametrize("kwargs", [{"shuffle": True}, {"drop_last": True}])
def test_reject_unaligned_loader(kwargs):
    loader = DataLoader(fixture_dataset(), batch_size=4, **kwargs)
    with pytest.raises(ValueError, match="sequential"):
        evaluation_indices(loader)


def test_reject_duplicate_ids():
    loader = DataLoader(Subset(fixture_dataset(), [0, 0]), batch_size=2)
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation_indices(loader)


def test_test_ids_with_synthetic_data(tmp_path):
    loader = DataLoader(fixture_dataset(), batch_size=4)
    rows = []
    result = evaluate_model(torch.nn.Identity(), loader, torch.device("cpu"), prediction_rows=rows)
    path = tmp_path / "test.csv"
    write_predictions(path, rows, "test", "dense")
    with path.open(newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert [r["image_id"] for r in exported] == [f"test:{i}" for i in range(6)]
    assert result["n"] == len(exported) == 6


def test_cli_export_validation(monkeypatch, tmp_path):
    loader = DataLoader(Subset(fixture_dataset(), [4, 2, 0]), batch_size=2)
    monkeypatch.setattr(evaluate, "load_dense_checkpoint", lambda *a: (torch.nn.Identity(), {}))
    monkeypatch.setattr(evaluate, "build_datasets_for_model", lambda *a, **kw: (None, loader, None, []))
    output, pred = tmp_path / "result.json", tmp_path / "pred.csv"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", "fake.pt", "--variant", "dense",
                                       "--output", str(output), "--predictions-output", str(pred)])
    evaluate.main()
    result = json.loads(output.read_text())
    assert result["scores"]["n"] == result["prediction_export"]["n"] == 3
    assert result["prediction_export"]["configuration"] == "dense"
    with pred.open(newline="") as stream:
        assert [r["image_index"] for r in csv.DictReader(stream)] == ["4", "2", "0"]


def test_cli_test_still_requires_frozen_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", "fake.pt", "--split", "test",
                                       "--variant", "dense", "--predictions-output", "unused.csv"])
    with pytest.raises(ValueError, match="Test is locked"):
        evaluate.main()


def test_cli_calibrated_label_validation(monkeypatch, tmp_path):
    from pathlib import Path
    config = Path("outputs/calibration_64_v4.json")
    loader = DataLoader(fixture_dataset(), batch_size=4)
    model = torch.nn.Identity()
    model.blocks = [None] * 12
    monkeypatch.setattr(evaluate, "load_dense_checkpoint", lambda *a: (model, {}))
    monkeypatch.setattr(evaluate, "sha256_file", lambda *a: json.loads(config.read_text())["checkpoint_sha256"])
    monkeypatch.setattr(evaluate, "build_datasets_for_model", lambda *a, **kw: (None, loader, None, []))
    calls = []
    def patch(model, mask_config, layer_indices, diagnostics):
        calls.append((mask_config, layer_indices, diagnostics))
        return []
    monkeypatch.setattr(evaluate, "patch_attention_layers", patch)
    output, pred = tmp_path / "result.json", tmp_path / "pred.csv"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", "fake.pt", "--frozen-config", str(config),
                                       "--label", "D_50", "--output", str(output), "--predictions-output", str(pred)])
    evaluate.main()
    result = json.loads(output.read_text())
    assert calls[0][1:] == ([3, 6, 9], True)
    assert result["mask_config"]["p"] == 0.65
    assert result["mask_config"]["top_k"] == 16
    assert result["mask_config"]["similarity_threshold"] == 0.50
    assert result["prediction_export"]["configuration"] == "D_50"
    with pred.open(newline="") as stream:
        assert {r["configuration"] for r in csv.DictReader(stream)} == {"D_50"}


def test_cli_rejects_same_json_and_csv_path(monkeypatch, tmp_path):
    path = str(tmp_path / "result.csv")
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", "fake.pt", "--output", path,
                                       "--predictions-output", path])
    with pytest.raises(ValueError, match="different paths"):
        evaluate.main()
