# Paired prediction analysis

Prepared 2026-11-26. This standalone analysis utility reads existing evaluator CSV/JSON pairs. It does not import a model, load images, change calibrated settings, or perform inference. The official test run remains gated on team approval of the freeze.

## Inputs and safeguards

Place each run's evaluator files together as `dense.csv`, `dense.json`, `A_50.csv`, `A_50.json`, etc. The utility validates the CSV schema, unique split-qualified IDs, official index ranges, breed IDs, correctness indicators, configuration/split metadata, row counts and aggregate accuracy. Validation IDs must belong to the original validation manifest. Rows are joined by image ID and ground truth must agree. Test analysis requires all ten named configurations and every official test index 0–3668 in each file. Raw SHA-256 fingerprints of the input files are saved with the analysis. These do not replace separately verifying the checkpoint hash, original dataset order/release, preprocessing, approved calibration and code revision: evaluator CSV/JSON files do not contain enough provenance to establish all of those conditions.

## Validation pilot on Windows PowerShell

From the repository root, reuse the existing interpreter and the saved 32-image validation pilot. No inference or dataset download is needed.

```powershell
$python = "$HOME\Downloads\SpargeFinalProject\sparge_pets_starter\.venv\Scripts\python.exe"
$pilot = "$env:TEMP\deit-prediction-pilot"
& $python -m sparse_pet.paired_analysis `
  --input-dir $pilot `
  --split val `
  --manifest outputs/split_manifest.json `
  --configurations dense A_50 `
  --repetitions 10000 `
  --seed 42 `
  --output "$pilot\paired_analysis.json"
```

The previous pilot produced 29 correct predictions out of 32 in each run, so the accuracy delta must be 0 percentage points. The interval and disagreement counts depend on individual predictions; equal accuracy alone does not determine these values. Pilot data are not class-balanced and are for functionality verification only.

## Final analysis (only after the approved official evaluation)

The complete official evaluator output directory must contain the ten CSV/JSON pairs for `dense`, `unmasked`, `A_25`, `B_25`, `C_25`, `D_25`, `A_50`, `B_50`, `C_50`, `D_50`. For example, if the team chooses `outputs/test_v4` as the evaluation directory:

```sh
python -m sparse_pet.paired_analysis --input-dir outputs/test_v4 --split test --repetitions 10000 --seed 42 --output outputs/test_v4/paired_analysis.json
```

This command analyzes already-generated predictions only; it does not run test inference. Use the approved final evaluation revision, and record hardware and package versions with the original evaluation outputs. Preserve all runs; do not select or retune variants using test outcomes.

## Statistical method and interpretation

For each candidate, form paired correctness differences `d_i = candidate_correct_i - dense_correct_i` in {-1, 0, +1}. The observed accuracy change in percentage points is `100 * mean(d_i)`. The analysis samples the empirical distribution of these paired differences with replacement, using multinomial counts for the three possible values. This is exactly equivalent in distribution to ordinary paired image bootstrap resampling for the mean correctness difference, without allocating a large matrix of sample indices. Use 10,000 replicates and seed 42 by default. The 2.5th and 97.5th percentiles form the marginal 95% percentile interval. Each comparison resets the RNG to the same seed for reproducibility; this is not a joint or simultaneous bootstrap across configurations.

Output fields include accuracy in [0,1], candidate-minus-dense delta and CI in percentage points, candidate-only correct (wins), dense-only correct (losses), both correct, both wrong, and predicted-label disagreement count/rate (rate in [0,1]). The latter includes changes between two incorrect labels. Wins minus losses determines accuracy delta. Equal accuracy may hide differing predictions. All-zero differences yield a degenerate [0,0] bootstrap interval; this is not evidence of equivalence. The utility does not implement equivalence testing, macro-F1 intervals, multiple-comparison corrections, or hypothesis-test p-values. Use the evaluator JSON for macro-F1. Intervals are descriptive/exploratory, assume independent sampled images, and are conditional on a single checkpoint; they do not measure training-seed or dataset-shift variability. Avoid declaring superiority from marginal intervals across multiple comparisons. No test-driven calibration changes.

## Verification

13 new synthetic tests plus the previous 38 tests passed (51 total) in the patch assembly CPU environment. Tests include shuffled row order, label mismatch, duplicate/missing IDs, correctness and JSON accuracy mismatch, manifest membership, deterministic bootstrap, CLI output, and the full ten-configuration 3669-row synthetic contract. No actual official test images or predictions were used. The user's saved 32-image pilot still needs to be analyzed on their PC to verify real exported files.
