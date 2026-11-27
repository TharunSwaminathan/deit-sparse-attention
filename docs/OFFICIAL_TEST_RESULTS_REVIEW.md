# Official test results and independent audit — November 27, 2026

**Status:** Numerical and archive audit passed. Final protocol acceptance remains conditional on documenting pre-run team approval and confirming dataset provenance. No inference rerun, retuning or training was performed in this audit.

## Provenance

- Archive: `official_test_v4_001.zip`; SHA-256 `5d304660067d89e0bb6ef8c5c7f73b24331f04600f1f2be09c40af5628c1030c` (matches notebook output).
- Source revision: `b1ebf0736004f8564c26803636467b451cf1ea16` (PR #1 head, included in merged main). The notebook used `original_v4`, not retraining.
- Frozen checkpoint SHA-256 (recorded): `ca12f33e158e30c08279ff8be5bd5b2a69be75e688a01eee59195fc3f198cb06`. Independently matched against the local original checkpoint; the submitted archive does not include weights.
- Archived calibration and split manifest canonical hashes independently match the freeze.
- Recorded hardware: Tesla T4; FP32; batch 16; workers 0; PyTorch 2.14.1+cu130; torchvision 0.29.1+cu130; timm 1.0.31; NumPy 2.4.6; scikit-learn 1.6.1.
- 224 × 224 evaluation, bicubic resize 248 and center crop, ImageNet normalization; layers 3, 6, 9 (zero-based).
- Run timestamps UTC: 2026-11-27T18:56:11.624055+00:00 to 2026-11-27T19:03:12.551074+00:00. Includes evaluation and paired analysis; not an inference latency benchmark.

## Accuracy and budget comparison

| Configuration | Validation accuracy (%) | Validation sparsity (%) | Test correct / 3669 | Test accuracy (%) | Test macro-F1 | Test sparsity (%) | Within ±3 pp? |
|---|---:|---:|---:|---:|---:|---:|---|
| dense | 93.75 | 0.00 | 3355 | 91.4418 | 0.912616 | 0.0000 | Yes |
| unmasked | 93.75 | 0.00 | 3355 | 91.4418 | 0.912616 | 0.0000 | Yes |
| A_25 | 93.89 | 23.74 | 3359 | 91.5508 | 0.913732 | 23.8177 | Yes |
| B_25 | 93.89 | 25.68 | 3359 | 91.5508 | 0.913695 | 25.7894 | Yes |
| C_25 | 93.61 | 23.98 | 3357 | 91.4963 | 0.913116 | 24.0178 | Yes |
| D_25 | 94.02 | 25.55 | 3355 | 91.4418 | 0.912589 | 25.7775 | Yes |
| A_50 | 93.75 | 49.85 | 3359 | 91.5508 | 0.913745 | 49.9148 | Yes |
| B_50 | 93.75 | 52.62 | 3346 | 91.1965 | 0.910059 | 53.1340 | **No** |
| C_50 | 93.89 | 50.17 | 3353 | 91.3873 | 0.912080 | 50.2350 | Yes |
| D_50 | 93.75 | 51.76 | 3347 | 91.2238 | 0.910278 | 52.0905 | Yes |

## Paired accuracy differences vs dense

| Configuration | Δ accuracy (pp) | Marginal 95% bootstrap CI (pp) | Candidate-only correct | Dense-only correct | Predicted class disagreements |
|---|---:|---|---:|---:|---:|
| unmasked | +0.000 | [+0.000, +0.000] | 0 | 0 | 0 |
| A_25 | +0.109 | [+0.000, +0.245] | 5 | 1 | 8 |
| B_25 | +0.109 | [+0.000, +0.245] | 5 | 1 | 8 |
| C_25 | +0.055 | [-0.082, +0.218] | 5 | 3 | 10 |
| D_25 | +0.000 | [-0.191, +0.191] | 7 | 7 | 16 |
| A_50 | +0.109 | [-0.191, +0.409] | 18 | 14 | 40 |
| B_50 | -0.245 | [-0.627, +0.136] | 21 | 30 | 60 |
| C_50 | -0.055 | [-0.354, +0.245] | 15 | 17 | 38 |
| D_50 | -0.218 | [-0.545, +0.136] | 16 | 24 | 46 |

All nine marginal intervals include zero (A_25 and B_25 touch zero). The unmasked interval is degenerate because all predictions match. These exploratory IID image-level percentile intervals (10,000 draws, seed 42) are not multiplicity-adjusted or equivalence tests and do not capture training-seed variability.

## Attention fidelity and ideal arithmetic savings

| Configuration | Retained mass (%) | Relative output error (%) | Ideal gross saved MFLOPs/image | All-layer attention product reduction (%) |
|---|---:|---:|---:|---:|
| dense | — | — | 0.00 | 0.00 |
| unmasked | 100.00 | 0.00 | 0.00 | 0.00 |
| A_25 | 98.33 | 2.84 | 21.30 | 5.95 |
| B_25 | 97.80 | 3.59 | 23.06 | 6.45 |
| C_25 | 96.93 | 5.17 | 21.48 | 6.00 |
| D_25 | 95.01 | 7.80 | 23.05 | 6.44 |
| A_50 | 88.39 | 15.50 | 44.63 | 12.48 |
| B_50 | 79.81 | 24.60 | 47.51 | 13.28 |
| C_50 | 88.53 | 15.53 | 44.92 | 12.56 |
| D_50 | 84.64 | 20.04 | 46.58 | 13.02 |

These ideal savings are computed as 3 × 4 × 197² × 192 × s FLOPs (s is the realized fraction), excluding mask generation and sparse-kernel overhead. The prototype still computes dense QKᵀ and AV, and the previous batch-1 T4 latency benchmark showed all sparse variants slower than native dense attention.

## Report-ready results and discussion

On the official 3,669-image Oxford-IIIT Pet test split, native dense DeiT-Tiny obtained 91.44% top-1 accuracy (3,355 correct) and macro-F1 0.9126. The manual unmasked control reproduced every predicted class. Across all eight frozen sparse configurations, accuracy ranged from 91.20% to 91.55%, corresponding to changes of −0.245 to +0.109 percentage points relative to dense. Seven of eight configurations remained within the prespecified ±3 percentage-point sparsity tolerance. B_50 reached 53.134% sparsity, slightly exceeding the 53% upper bound; this result is retained without recalibration. A_50 achieved 49.915% sparsity and 91.55% accuracy (3,359 correct), retaining 88.39% of dense attention mass with mean relative output error of 15.50%. Its paired accuracy change was +0.109 pp (marginal 95% CI −0.191 to +0.409 pp). All nine comparisons against dense had intervals including zero. Accordingly, the findings support small observed accuracy changes at substantial attention sparsity for this checkpoint, but not statistical superiority or equivalence.

At the higher sparsity target, A_50 and C_50 showed similar attention fidelity (retained mass 88.39% and 88.53%; relative output error 15.50% and 15.53%). B_50 and D_50 had higher realized sparsity and lower retained mass, alongside larger relative errors. The protected variants did not consistently improve aggregate fidelity or accuracy under the selected global budgets. This comparison does not isolate the causal effect of similarity protection because the thresholds, top-k settings, and realized budgets differ. The earlier validation block diagnostics indicate layer-dependent relationships; these were not recomputed on test images.

For A_50, ideal gross block skipping would save 44.63 MFLOPs per image, or 12.48% of attention matrix-product arithmetic across all twelve layers (about 1.78% of the restricted full-model matrix-product estimate). This is hypothetical rather than measured acceleration. Our current dense-matrix masking implementation does not skip these products.

## Independent audit and open checks

- Verified ZIP CRC, uniqueness of all paths and all 50 manifest file hashes; the 22 raw result files are present.
- Verified all 20 prediction/aggregate hashes against paired analysis and run provenance; verified paired output hash. Archive has 51 entries including its own checksum manifest.
- Verified all 36,690 rows using the existing paired-analysis reader: complete official index range, matching class labels, valid classes and correctness, unique IDs, and CSV/JSON accuracy agreement.
- Independently recomputed macro-F1 over all 37 classes and reran the existing paired analysis under NumPy 2.3.5; all comparison fields, including all confidence intervals, exactly reproduced the delivered values. Checked selected mask parameters and layer indices against the frozen calibration file.
- Verified derived summary accuracy, loss, macro-F1, sparsity, paired comparison fields, tolerance flags and ideal gross FLOP savings against raw records/formula. Loss and attention diagnostics cannot themselves be recomputed from class-prediction CSVs.
- Recorded logs show 56 passing tests and native/unmasked logit maximum absolute difference 7.15e-7 on the checkpoint smoke test; these are Jiong’s execution records, not independently rerun GPU tests.
- Jiong confirmed on November 27 that he used the pre-run experimental freeze. The approval reference is only `Team approval`; notebook sets `APPROVED_REVISION = REVISION`, `RUN_OFFICIAL_TEST = True`, `CHECKOUT_COMMIT = ""`. Obtain the actual pre-run approval record (date, participants and agreed revision) from team messages. If none exists, disclose that the run preceded documented approval; do not backdate or retune. The official test results have already been viewed.
- Dev should confirm the actual downloaded official dataset release and preprocessing. Consistent test indices/labels do not by themselves authenticate image contents. No independent model inference was performed in this audit.
- Before committing the notebook as reusable code, set execution disabled by default and require an explicit pinned revision and approval reference. Preserve the original executed notebook and archive unchanged as evidence.

## Figure handoff

Jiong already included `derived/test_accuracy_vs_sparsity.{png,pdf}`, `derived/test_accuracy_delta_ci.{png,pdf}` and `derived/ideal_attention_flops.{png,pdf}`. Use these existing figures with captions that identify the official test set and the hypothetical selected-layer FLOP denominator. Do not present test results as a new tuning criterion.

### November 27 follow-up

Jiong also confirmed the Colab checkpoint location `checkpoints/best.pt` inside the Drive project folder. The reported freeze use is consistent with the archived hashes and configuration settings; it does not identify an explicit dated team approval. Keep this distinction in the final report. No rerun is needed to prepare these documents.
