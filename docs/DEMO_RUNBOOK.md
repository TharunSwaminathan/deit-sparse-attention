# Live demonstration runbook — When to Trust Block Pooling?

**Prepared:** November 27, 2026. **Scope:** validation-only demonstration; no official test inference. Use merged revision `2915614fdcd4e219b7c039a96cdb8e073f48bc11` or a team-verified descendant with unchanged inference and frozen inputs. The demo reuses the existing evaluator and paired-analysis module; no model changes, training, calibration, benchmark or internet download is required.

## Goal and timing

Aim for a 90–120 second demonstration after the implementation slides, subject to rehearsal and the team's 12–15 minute total slot. Show the architecture (10–15 s), start a dense versus A_50 run (use the execution time to explain the mask), then show the compact summary and paired predictions (20–30 s). Use the fallback if inference takes too long. Do not interpret elapsed demo time as inference latency.

## Rehearsal / live run — Windows PowerShell 5.1

Run from the repository root. The paths below are for the original implementation contributor's PC; teammates should substitute their own local paths. Use the locally extracted Oxford-IIIT Pet dataset and privately shared full fine-tuned checkpoint. The official launcher preflight should already have passed for this revision; run it again in dry-run mode if the revision or inputs change.

```powershell
cd "$HOME\Downloads\deit-sparse-attention"
$python = "$HOME\Downloads\SpargeFinalProject\sparge_pets_starter\.venv\Scripts\python.exe"
$checkpoint = "$HOME\Downloads\SpargeFinalProject\sparge_pets_starter\outputs\dense_full_pilot\best.pt"
$dataRoot = "$HOME\Downloads\SpargeFinalProject\sparge_pets_starter\data\downloads"
$manifest = "outputs/split_manifest.json"
$frozen = "outputs/calibration_64_v4.json"
$demo = Join-Path "$HOME\Downloads" ("deit-demo-" + [guid]::NewGuid().ToString("N").Substring(0,8))

foreach ($path in @($python, $checkpoint, $dataRoot, $manifest, $frozen)) {
    if (-not (Test-Path $path)) { throw "Missing input: $path" }
}
if ((Get-FileHash $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant() -ne "ca12f33e158e30c08279ff8be5bd5b2a69be75e688a01eee59195fc3f198cb06") {
    throw "Checkpoint fingerprint mismatch"
}
New-Item -ItemType Directory -Path $demo -ErrorAction Stop | Out-Null
$common = @("-m", "sparse_pet.evaluate", "--checkpoint", $checkpoint, "--data-root", $dataRoot, "--manifest", $manifest, "--frozen-config", $frozen, "--split", "val", "--batch-size", "16", "--workers", "0", "--max-images", "32")

& $python @common --variant dense --output "$demo\dense.json" --predictions-output "$demo\dense.csv"
if ($LASTEXITCODE -ne 0) { throw "Dense demo failed" }
& $python @common --label A_50 --output "$demo\A_50.json" --predictions-output "$demo\A_50.csv"
if ($LASTEXITCODE -ne 0) { throw "Sparse demo failed" }
& $python -m sparse_pet.paired_analysis --input-dir $demo --split val --manifest $manifest --configurations dense A_50 --repetitions 10000 --seed 42 --output "$demo\paired_analysis.json"
if ($LASTEXITCODE -ne 0) { throw "Paired analysis failed" }

$d = Get-Content "$demo\dense.json" -Raw | ConvertFrom-Json
$a = Get-Content "$demo\A_50.json" -Raw | ConvertFrom-Json
$p = Get-Content "$demo\paired_analysis.json" -Raw | ConvertFrom-Json
if ($d.split -ne "val" -or $a.split -ne "val" -or $p.split -ne "val" -or $d.scores.n -ne 32 -or $a.scores.n -ne 32 -or $p.n -ne 32) { throw "Unexpected demo scope" }
$summary = @(
    [pscustomobject]@{Config="dense"; Images=$d.scores.n; AccuracyPct=[math]::Round(100*$d.scores.accuracy,2); SparsityPct=0; RetainedMassPct="N/A"},
    [pscustomobject]@{Config="A_50"; Images=$a.scores.n; AccuracyPct=[math]::Round(100*$a.scores.accuracy,2); SparsityPct=[math]::Round(100*$a.diagnostics.realized_sparsity,2); RetainedMassPct=[math]::Round(100*$a.diagnostics.retained_mass,2)}
)
$summary | Format-Table -AutoSize
$p.comparisons.A_50 | Format-List accuracy_delta_pp, prediction_disagreements, candidate_only_correct, dense_only_correct, both_correct, both_wrong
Write-Host "DEMO COMPLETE (VALIDATION ONLY): $demo"
```

## Reference rehearsal evidence (November 25–26)

The earlier 32-image validation pilot returned 29/32 correct for both dense and A_50 (90.625%); A_50 realized sparsity 50.3652%, retained mass 88.2855%, relative output error 15.8737%. Paired analysis on November 26 found zero prediction disagreements. This is reference evidence, not a guarantee of identical future results. The first 32 ordered validation indices are not class-balanced, so the pilot is a functionality demonstration only. A_50 uses Top-p p=0.87, layers 3,6,9 (zero-based), with all CLS edges retained. No claim of equivalence or acceleration.

## Suggested narration

"This is our fixed validation subset, not the held-out test set. Both configurations use the same trained weights and image order. Dense keeps all attention connections; A_50 uses pooled query and key blocks to select which connections to retain in layers 3, 6 and 9. We then compare the predictions for the same images. The sparsity is measured in the modified attention layers, not across the entire model. The full validation experiment is more representative than this small demo. Our T4 benchmark shows masking overhead rather than acceleration because score and value products remain dense."

## Offline fallback

Before the presentation, open `figures/01_attention_architecture.png`, `figures/02_accuracy_vs_sparsity.png` and `figures/04_t4_latency.png`, and keep the latest successful demo output folder accessible. If execution fails or exceeds the time budget, stop the live demo and explicitly label the fallback as previously recorded results. Do not rerun the official test or download data in class. To show the full-validation records offline:

```powershell
Get-Content .\outputs\validation_v4\dense.json -Raw
Get-Content .\outputs\validation_v4\A_50.json -Raw
```

These two full-validation records each cover 736 images, both report 93.75% accuracy, and A_50 reports 49.8482% realized sparsity and 88.3941% retained mass. They are aggregate results, not evidence of identical predictions on all 736 images. Use the saved paired pilot only for per-image agreement.

## Acceptance and limitations

- Rehearse on the presentation PC with internet disconnected if practical; the demo should need only local files. Confirm both evaluation commands exit successfully and paired analysis completes on exactly 32 validation images.
- Check the A_50 mask configuration, selected layers and measured sparsity in its JSON; compare results with the reference pilot but report actual values if they differ. Record actual elapsed rehearsal time and choose whether live inference fits the presentation slot.
- The evaluator constructs a test dataset object during setup but selects only the validation loader for inference. This does not perform official test evaluation.
- Do not commit checkpoint, dataset or transient demo outputs to the repository; optionally commit this runbook after successful rehearsal. This runbook's CLI arguments were checked against saved source files; the Windows command sequence has not been executed in the authoring environment.
