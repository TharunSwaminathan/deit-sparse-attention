# DeiT-Tiny Sparse Attention — Oxford-IIIT Pet

Course final project: compare Top-p, similarity-protected Top-p, hybrid Top-k/Top-p, and protected hybrid masks at matched attention sparsity.

## Current status (2026-11-24)

- V4 reference implementation; four variants, eight calibrated configurations; zero-based attention blocks 3, 6, 9.
- Full 736-image validation and Tesla T4 latency measurements completed. Dense baseline validation accuracy: 93.75%. Official held-out test evaluation is pending team freeze.
- This prototype computes dense QK scores and masks them; GPU speedup was not observed.

## Setup

```sh
python -m pip install -r requirements.txt
python -m pytest -q tests
```

The dataset and trained checkpoint are intentionally not tracked. Obtain `Sparge_Project_Backup.zip` privately from the team. Restore `outputs/dense_full_pilot/best.pt`; its SHA-256 must be `ca12f33e158e30c08279ff8be5bd5b2a69be75e688a01eee59195fc3f198cb06`. The split manifest and calibrated settings are already tracked. Do not retrain or tune on the official test set.

## Repository contents

- `sparse_pet/`: V4 source code (unchanged from validated archive).
- `tests/`: V4 tests.
- `outputs/`: small calibration, mask overlap, validation, split manifest, layer probe and GPU timing JSON files.
- [Final validation and GPU handoff](docs/Tharun_Final_GPU_Benchmark_and_Validation_Handoff.md): latest experimental results and limitations.
- [Combined result table](docs/Tharun_Final_Validation_and_T4_GPU_Results.csv).
- [Original V4 handoff](THARUN_HANDOFF.md).
- [Historical starter instructions](docs/STARTER_README.md), [historical Colab instructions](RUN_NEXT_COLAB.md), and [team kickoff](TEAM_KICKOFF.md). These retain earlier version labels and pilot commands for reference; use the current calibrated settings and latest results instead of repeating old experiments.

## Team next steps

Review and freeze the checkpoint, layers and calibration configurations; complete official held-out test evaluation and final analysis.
