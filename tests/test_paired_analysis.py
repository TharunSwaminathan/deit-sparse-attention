import csv
import json

import pytest

from sparse_pet.paired_analysis import COLUMNS, analyze, paired_comparison, read_run


def write_run(root, name, predictions, split="val"):
    prefix = "trainval" if split == "val" else "test"
    fields = ["split", "configuration", "image_id", "image_index", "ground_truth", "predicted_class", "correct"]
    assert set(fields) == COLUMNS
    with (root / f"{name}.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(fields)
        for i, y, p in predictions:
            w.writerow([split, name, f"{prefix}:{i}", i, y, p, int(y == p)])
    (root / f"{name}.json").write_text(json.dumps({"split": split, "variant": name.split("_")[0],
        "prediction_export": {"configuration": name, "n": len(predictions), "id_scheme": "official_split:zero_based_index"},
        "scores": {"n": len(predictions), "accuracy": sum(y == p for _, y, p in predictions)/len(predictions)}}))


@pytest.fixture
def runs(tmp_path):
    write_run(tmp_path, "dense", [(1, 0, 0), (2, 1, 0), (3, 2, 0), (4, 0, 0)])
    write_run(tmp_path, "A_50", [(4, 0, 1), (2, 1, 1), (1, 0, 0), (3, 2, 2)])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"val_indices": [1, 2, 3, 4]}))
    return tmp_path, manifest


def test_alignment_and_counts(runs):
    root, manifest = runs
    result = analyze(root, "val", ["dense", "A_50"], manifest)
    r = result["comparisons"]["A_50"]
    assert r["n"] == 4 and r["dense_accuracy"] == .5 and r["candidate_accuracy"] == .75
    assert r["accuracy_delta_pp"] == 25 and r["candidate_only_correct"] == 2 and r["dense_only_correct"] == 1
    assert r["both_correct"] == 1 and r["both_wrong"] == 0 and r["prediction_disagreements"] == 3
    assert r["ci95_delta_pp"][0] <= 25 <= r["ci95_delta_pp"][1]
    assert result == analyze(root, "val", ["dense", "A_50"], manifest)


def test_identical_correctness_not_identical_predictions():
    a = {"a": (0, 1, 0), "b": (1, 1, 1)}
    b = {"b": (1, 1, 1), "a": (0, 2, 0)}
    r = paired_comparison(a, b)
    assert r["ci95_delta_pp"] == [0, 0] and r["degenerate_bootstrap"]
    assert r["prediction_disagreements"] == 1


def test_missing_ids(runs):
    root, manifest = runs
    write_run(root, "A_50", [(1, 0, 0)])
    with pytest.raises(ValueError, match="ID sets"):
        analyze(root, "val", ["dense", "A_50"], manifest)


def test_ground_truth_mismatch(runs):
    root, manifest = runs
    write_run(root, "A_50", [(1, 1, 0), (2, 1, 1), (3, 2, 2), (4, 0, 1)])
    with pytest.raises(ValueError, match="Ground-truth"):
        analyze(root, "val", ["dense", "A_50"], manifest)


def test_duplicate_ids(runs):
    root, _ = runs
    write_run(root, "A_50", [(1, 0, 0), (1, 0, 0)])
    with pytest.raises(ValueError, match="duplicate"):
        read_run(root, "A_50", "val")


def test_wrong_correctness(runs):
    root, _ = runs
    path = root / "dense.csv"
    path.write_text(path.read_text().replace("trainval:1,1,0,0,1", "trainval:1,1,0,0,0"))
    with pytest.raises(ValueError, match="correctness"):
        read_run(root, "dense", "val")


def test_json_accuracy_mismatch(runs):
    root, _ = runs
    path = root / "dense.json"
    obj = json.loads(path.read_text()); obj["scores"]["accuracy"] = 1
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="accuracy mismatch"):
        read_run(root, "dense", "val")


def test_manifest_membership(runs):
    root, manifest = runs
    manifest.write_text(json.dumps({"val_indices": [1, 2, 3]}))
    with pytest.raises(ValueError, match="outside"):
        analyze(root, "val", ["dense", "A_50"], manifest)


def test_test_requires_full_ids(tmp_path):
    write_run(tmp_path, "dense", [(0, 0, 0)], split="test")
    with pytest.raises(ValueError, match="3669"):
        read_run(tmp_path, "dense", "test")
    with pytest.raises(ValueError, match="ten configurations"):
        analyze(tmp_path, "test", ["dense", "A_50"])


def test_wrong_configuration_metadata(runs):
    root, _ = runs
    path = root / "A_50.json"
    obj = json.loads(path.read_text()); obj["prediction_export"]["configuration"] = "B_50"
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="metadata"):
        read_run(root, "A_50", "val")


def test_bootstrap_all_wins():
    a = {str(i): (0, 1, 0) for i in range(8)}
    b = {str(i): (0, 0, 1) for i in range(8)}
    r = paired_comparison(a, b)
    assert r["accuracy_delta_pp"] == 100 and r["ci95_delta_pp"] == [100, 100]


def test_cli(runs, monkeypatch, capsys):
    from sparse_pet.paired_analysis import main
    root, manifest = runs
    output = root / "paired.json"
    monkeypatch.setattr("sys.argv", ["paired_analysis", "--input-dir", str(root), "--manifest", str(manifest),
        "--configurations", "dense", "A_50", "--output", str(output)])
    main()
    assert "PAIRED ANALYSIS COMPLETE" in capsys.readouterr().out
    assert json.loads(output.read_text())["comparisons"]["A_50"]["accuracy_delta_pp"] == 25


def test_full_test_contract(tmp_path):
    from sparse_pet.paired_analysis import CONFIGURATIONS
    for name in CONFIGURATIONS:
        write_run(tmp_path, name, [(i, i % 37, i % 37) for i in range(3669)], split="test")
    result = analyze(tmp_path, "test", repetitions=1000)
    assert result["n"] == 3669 and len(result["comparisons"]) == 9
    assert all(r["accuracy_delta_pp"] == 0 for r in result["comparisons"].values())
