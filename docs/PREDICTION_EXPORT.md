# Optional aligned prediction export

**Prepared 2026-11-25.** This follow-up to V4 adds an opt-in CSV export to `python -m sparse_pet.evaluate`. It does not modify attention masks, calibrated settings, model weights, preprocessing, or split selection. Team approval of the proposed experimental freeze remains pending.

## Usage

Add `--predictions-output outputs/prediction_pilot/dense.csv` to the existing evaluation command. Keep the aggregate JSON (`--output`) and CSV paths distinct. Without the optional flag, the original aggregate-only behavior is retained.

Example validation-only pilot (requires the privately shared checkpoint, original manifest, Oxford-IIIT Pet data, and project dependencies):

```sh
python -m sparse_pet.evaluate --checkpoint outputs/dense_full_pilot/best.pt --split val --variant dense --frozen-config outputs/calibration_64_v4.json --batch-size 16 --workers 0 --max-images 16 --output outputs/prediction_pilot/dense.json --predictions-output outputs/prediction_pilot/dense.csv
python -m sparse_pet.evaluate --checkpoint outputs/dense_full_pilot/best.pt --split val --frozen-config outputs/calibration_64_v4.json --label A_25 --batch-size 16 --workers 0 --max-images 16 --output outputs/prediction_pilot/A_25.json --predictions-output outputs/prediction_pilot/A_25.csv
```

Use the original data-root via `--data-root` if it is not under `data/downloads`. Do not overwrite existing validation results. Pilots are for verifying export, not retuning. Existing dataset loader constructs a test dataset object but these commands evaluate validation images only.

## CSV schema

| Column | Meaning |
|---|---|
| `split` | `val` or `test` |
| `configuration` | Exact calibrated label, e.g. `A_25`, or control `dense`/`unmasked`; manual validation uses the variant name |
| `image_id` | `trainval:<index>` for validation or `test:<index>` for test |
| `image_index` | Zero-based underlying official dataset index |
| `ground_truth` | Zero-based breed label |
| `predicted_class` | Zero-based argmax class |
| `correct` | 1 for a correct prediction, 0 otherwise |

IDs refer to annotation-order indices, not image filenames or content hashes. The validation subset is mapped back to the original trainval indices, rather than numbering rows from zero. Pairing requires the same dataset release/order and original split manifest. Export rejects shuffled/nonstandard samplers, custom collate functions, dropping batches, out-of-order loading, and duplicate subset indices. Standard project validation/test loaders are sequential.

## Before paired analysis

For each configuration, confirm unique IDs, identical ID sets and matching ground-truth labels across runs, and that the CSV row count and mean correctness agree with aggregate `scores.n` and `scores.accuracy`. For full official test, expect 3,669 rows per configuration. Join by `image_id`, never by row position alone. Use paired resampling of image IDs for accuracy differences; do not claim statistical superiority from one or two extra correct images. Retain aggregate macro-F1 over 37 classes.

## Official test (only after team approval)

Use the same frozen checkpoint and calibration, full test split, and no `--max-images`. For the controls specify `--variant dense` or `--variant unmasked`; for each of the eight sparse configurations specify its `--label` and `--frozen-config outputs/calibration_64_v4.json`. The controls also require the frozen config. Use distinct JSON and CSV paths for each of the ten runs. The existing test guard checks for a frozen calibration and disallows partial test evaluation; it does not verify human team approval. Confirm approval, dataset/manifest identity and exact code revision separately before running. Report actual test sparsity without test-based recalibration.

## Verification status

Original 25 tests plus 10 new tests (35 total) passed in a CPU environment with synthetic data. Tests cover metric preservation, stable subset and test indices, partial batches, unsafe samplers, duplicates, CLI exports, calibrated label resolution, and existing test guard. No official test images, real model checkpoint inference, or real-data export were evaluated in this verification. Run the full suite on the team's environment and a validation-only pilot before the official run.
