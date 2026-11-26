# Experimental configuration freeze — V4

**Prepared:** 2026-11-25  
**Status:** Proposed freeze — pending Dev and Jiong confirmation. Do not run official test evaluation until approved.  
**Reference implementation:** `04295ef` on `tharun/sparse-attention` (PR #1).

## Fixed inputs

- Model: `deit_tiny_patch16_224.fb_in1k`, 37 classes. Load full fine-tuned weights before wrapping attention; no sparse retraining.
- Checkpoint: `outputs/dense_full_pilot/best.pt` (shared privately). Raw SHA-256: `ca12f33e158e30c08279ff8be5bd5b2a69be75e688a01eee59195fc3f198cb06`. Local Windows copy verified on November 25; teammate copies still need verification.
- Dataset: Oxford-IIIT Pet. Use the existing `outputs/split_manifest.json`: 2,944 train / 736 validation (seed 42, stratified 20% validation). Reserve the official 3,669-image test split for final evaluation. Do not regenerate the manifest.
- Preprocessing: existing timm model-resolved evaluation transform (`resolve_model_data_config`, `create_transform(..., is_training=False)`); 224 × 224 images. Record exact resolved transform and package versions at final evaluation.
- Attention layers: 3, 6, 9 (zero-indexed), chosen after validation probes to cover depth; not claimed optimal.
- 14 × 14 patch grid; 2 × 2 spatial pooling; 49 coarse blocks; CLS row and column always retained. Apply mask before token-level softmax.
- Controls: native dense and manual unmasked wrapper on blocks 3, 6, 9, using the same weights.

## Selected sparse configurations

Authoritative file: `outputs/calibration_64_v4.json` (64 validation images). All eight configurations met the ±3 percentage-point tolerance on calibration and full validation. Stored k=4 is inactive for A and B.

| Label | Method | p | k (stored) | Similarity threshold | Calibration sparsity | Full validation sparsity |
|---|---|---:|---:|---:|---:|---:|
| A_25 | Top-p | 0.98 | 4 | — | 24.32% | 23.74% |
| B_25 | Protected Top-p | 0.97 | 4 | 0.5 | 25.84% | 25.68% |
| C_25 | Top-k union Top-p | 0.95 | 32 | — | 24.30% | 23.98% |
| D_25 | Protected hybrid | 0.65 | 32 | 0.55 | 24.67% | 25.55% |
| A_50 | Top-p | 0.87 | 4 | — | 50.45% | 49.85% |
| B_50 | Protected Top-p | 0.65 | 4 | 0.55 | 50.80% | 52.62% |
| C_50 | Top-k union Top-p | 0.85 | 12 | — | 50.75% | 50.17% |
| D_50 | Protected hybrid | 0.65 | 16 | 0.5 | 50.34% | 51.76% |

## Reproducibility fingerprints

The following are SHA-256 hashes of parsed JSON serialized using `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")`. These avoid Windows CRLF differences; they are not raw file hashes.

- `outputs/calibration_64_v4.json`: `c04ebcc0534cf814f660be6357479d44c948a65f6daa0e8d631274915201d0e4`
- `outputs/split_manifest.json`: `a0666d0cb771f7ce8ee0f45600d099b68d412bf8bf76cfeddca25c7fbc3b0b33`

## Proposed final evaluation protocol (team confirmation required)

- Evaluate all ten configurations on the same full official test split, without shuffling, training or test-based parameter selection. Use FP32 and inference mode, proposed batch size 16. Record hardware and exact torch, torchvision, timm, numpy and scikit-learn versions.
- Save aggregate accuracy, macro-F1 over 37 classes, cross-entropy, sample count and attention diagnostics. Save aligned per-image sample indices, labels and predictions for paired accuracy comparisons; the optional prediction-export follow-up is documented in [PREDICTION_EXPORT.md](PREDICTION_EXPORT.md) and must pass integration verification before the official run.
- Report actual test sparsity even if outside ±3 percentage points of target. Do not recalibrate on test. Compare budget-matched variants, not isolated causal effects with all other parameters fixed.
- Existing GPU timing: synthetic GPU-resident FP32 inputs, batch 1, 10 warmups and 30 measured repetitions, synchronized CUDA, median, diagnostics off. This prototype computes dense scores and value products; no sparse kernel or measured speedup. Attention-edge sparsity is not total model FLOP reduction.
- Remaining diagnostics: evaluate relationships between similarity, entropy and attention error using validation data. Define analysis before viewing test outcomes. Existing aggregate diagnostics are not per-image distributions.
- Record the final approved code revision and approval date. Prediction-export changes must not change mask/model logic; document any later bug fixes. No test-driven hyperparameter retuning.

## Approval checklist

- [x] Local checkpoint hash verified
- [x] All eight full-validation sparsity measurements within tolerance
- [x] V4 unit tests passed locally (25/25)
- [ ] Dev verifies checkpoint, split and preprocessing
- [ ] Jiong confirms evaluation protocol and prediction export
- [ ] PR review and team agreement on frozen configurations
- [ ] Record approval date and exact evaluation revision before official test run
