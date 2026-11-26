# Official test workflow — proposed V4 protocol

**Do not execute until the team approves the exact evaluation revision and configuration freeze.** Jiong is the evaluation lead; Dev independently verifies checkpoint, dataset and preprocessing. The launcher is an additional reproducibility utility, not a change to the masking or evaluation algorithms.

## Fixed inputs and outputs

Use the privately shared full fine-tuned checkpoint (`best.pt`), `outputs/calibration_64_v4.json`, and `outputs/split_manifest.json`. The launcher verifies the raw checkpoint SHA-256 and the canonical JSON SHA-256 fingerprints documented in `EXPERIMENT_FREEZE.md`. It does not accept manual sparsity parameters. It uses batch 16, default workers 0, native dense and manual unmasked controls, and all eight calibrated sparse variants on the entire official 3,669-image test split. No training, tuning, max-images limit or latency benchmark is included. GPU recommended; CPU requires explicit `--allow-cpu` at execution. Keep batch size and environment fixed across configurations.

## 1. Dry run (safe before approval)

From the repository root in Windows PowerShell:

```powershell
$python = "$HOME\Downloads\SpargeFinalProject\sparge_pets_starter\.venv\Scripts\python.exe"
$checkpoint = "$HOME\Downloads\SpargeFinalProject\sparge_pets_starter\outputs\dense_full_pilot\best.pt"
$dataRoot = "$HOME\Downloads\SpargeFinalProject\sparge_pets_starter\data\downloads"
$testOut = "$HOME\Downloads\deit-official-test-v4"
& $python scripts/run_official_test.py --checkpoint $checkpoint --data-root $dataRoot --output-dir $testOut
```

Expected: ten evaluator commands plus the paired analysis command, matching hashes, and `PREFLIGHT PASS`. The preflight does not load model weights, instantiate a dataset, create output files, or perform inference. It can run on a dirty feature branch, and reports HEAD and its clean/dirty status. It does not verify that the dataset is available, GPU is working, or the environment has all dependencies. It is not team approval.

## 2. After actual team approval only

Obtain the exact approved full Git SHA and the dated approval reference from the team. Use a clean checkout of that revision (do not assume a new merge SHA is approved). Install `requirements.txt` with a compatible GPU PyTorch environment, run `python -m pytest -q`, and verify dataset release and evaluation transforms with Dev. Use a new output directory **outside the repository**. Preserve all ten runs together. For Windows PowerShell, reuse the variables above and explicitly replace these two placeholders:

```powershell
$approvedRevision = "REPLACE_WITH_APPROVED_FULL_SHA"
$approvalReference = "REPLACE_WITH_ACTUAL_DATED_TEAM_APPROVAL_REFERENCE"
& $python scripts/run_official_test.py --checkpoint $checkpoint --data-root $dataRoot --output-dir $testOut --execute --approved-revision $approvedRevision --approval-reference $approvalReference
```

On Colab, run from the approved repository root, with the real checkpoint and dataset paths (example placeholders):

```bash
python scripts/run_official_test.py --checkpoint /content/shared/best.pt --data-root /content/pets-data --output-dir /content/deit-official-test-v4 --execute --approved-revision APPROVED_FULL_SHA --approval-reference 'ACTUAL DATED TEAM APPROVAL REFERENCE'
```

In a Colab notebook prefix the command with `!`. Add `--download` if the official dataset has not been downloaded; the launcher only forwards this flag when executing. Copy the completed output off the temporary Colab runtime. Do not commit private weights or raw datasets to GitHub.

## 3. Validate and hand off

A complete run creates ten aggregate JSON files and ten CSVs, `paired_analysis.json` and `run_provenance.json` (22 files). Expect 3,669 unique test IDs in each CSV; all ground-truth labels must match across runs. The launcher checks configuration metadata, sample counts, CSV/JSON accuracy consistency, device consistency, frozen input hashes and approved Git revision. The paired-analysis module validates matching IDs and labels for all ten configurations and generates marginal paired bootstrap intervals (10,000 repetitions, seed 42). `run_provenance.json` records the commands, approval reference, exact package versions, hardware, resolved preprocessing, output fingerprints and completion status. Inspect the dense/unmasked prediction agreement as an implementation control. Investigate material discrepancies before interpreting results; document any necessary bug fixes and obtain renewed approval without tuning on test.

Report achieved test sparsity even when it falls outside tolerance; do not recalibrate. The paired intervals are exploratory, unadjusted for multiplicity, and conditional on one trained checkpoint. Degenerate intervals do not prove equivalence. GPU timings are separate from this workflow.

If a run fails, keep the partial directory and failure record; do not mix results from multiple executions or overwrite files. Diagnose the cause and coordinate a fresh complete run in a new directory. No automatic resume is implemented.

## Limitations

The approval fields are a human-controlled guard, not independent proof of approval or a security boundary. The launcher does not fingerprint the entire dataset or verify every image's contents; Dev must verify dataset provenance separately. Package versions and resolved transform are recorded once at execution start, not pinned automatically. Git clean status does not cover ignored files. Do not modify software, inputs or environment during evaluation. The launcher checks input/configuration hashes and outputs but cannot ensure identical floating-point behavior across hardware. Only dry-run and unit-test behavior has been verified before approval; the full held-out workflow remains unexecuted.
