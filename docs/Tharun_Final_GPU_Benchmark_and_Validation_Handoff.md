# Sparse Attention DeiT-Tiny — Validation and Tesla T4 GPU Handoff

**Date:** 2026-11-23  
**Scope:** Tharun’s sparse attention implementation / full validation / inference timing  
**Model:** DeiT-Tiny trained on Oxford-IIIT Pet (37 classes)  
**Checkpoint SHA-256:** `ca12f33e158e30c08279ff8be5bd5b2a69be75e688a01eee59195fc3f198cb06`  
**Selected Transformer blocks:** 3, 6, 9 (zero based)  
**Calibration:** 64 validation images; eight configs within ±3 percentage points of 25% and 50% target sparsity.  
**Full validation:** 736 images; held-out test set not used.  
**GPU:** Tesla T4 | PyTorch 2.14.1+cu130 | FP32 | batch size 1 | 10 warmup and 30 measured repetitions.

## Results: full validation + synthetic-input GPU forward timing

| Method | Realized sparsity | Accuracy | Macro-F1 | GPU median (ms) | vs dense |
|---|---:|---:|---:|---:|---:|
| dense | 0.00% | 93.75% | 0.9379 | 4.739 | 1.00× |
| unmasked | 0.00% | 93.75% | 0.9379 | 5.235 | 1.10× |
| A_25 | 23.74% | 93.89% | 0.9393 | 7.327 | 1.55× |
| B_25 | 25.68% | 93.89% | 0.9393 | 8.701 | 1.84× |
| C_25 | 23.98% | 93.61% | 0.9366 | 7.570 | 1.60× |
| D_25 | 25.55% | 94.02% | 0.9406 | 9.171 | 1.94× |
| A_50 | 49.85% | 93.75% | 0.9379 | 7.833 | 1.65× |
| B_50 | 52.62% | 93.75% | 0.9379 | 11.244 | 2.37× |
| C_50 | 50.17% | 93.89% | 0.9393 | 10.761 | 2.27× |
| D_50 | 51.76% | 93.75% | 0.9379 | 9.275 | 1.96× |

## Interpretation

- Full validation accuracy was tightly clustered across dense and sparse variants. For example, D at 25% reached 94.02% versus 93.75% dense: two more correct predictions out of 736, not evidence of statistically significant superiority.
- **No GPU speedup was observed.** All sparse variants were 1.55×–2.37× slower than native dense inference. This implementation calculates full dense attention, then constructs and applies masks, without a sparse CUDA kernel.
- Native dense forward time was 4.739 ms; unmasked attention wrapper was 5.235 ms, roughly 10.5% overhead even when no connections were removed.
- The attention sparsity percentages apply to connections in selected blocks 3, 6, 9, **not** 25% or 50% overall model FLOP reduction. The reported timings include mask overhead and exclude preprocessing/host-to-device transfer.
- The selected settings are matched for sparsity but differ in Top-p and Top-k values. Thus comparisons measure **budget-matched implementations**, not isolated causal effects with all other hyperparameters fixed.
- Calibration and mask-overlap checks used the same dense checkpoint, and pairwise mask differences were nonzero in all 12 comparisons, but mask-overlap measured on identical dense Q/K inputs is a separate diagnostic from full sparse forward passes.

## Method and measurement notes

The GPU timing function constructs one random GPU-resident input of shape (1,3,224,224), runs model forward calls in `torch.inference_mode()`, synchronizes CUDA immediately before and after each iteration, discards 10 warm-up measurements, and reports the median of the next 30 measurements. The results contain only medians, so run-to-run uncertainty and repeatability across sessions are not established. No inference speedup should be claimed.

## Recommended handoff / next tasks

1. Commit Version 4 source, tests, and documented settings to the shared final-project repository, excluding model weights.
2. Give the team the calibration JSON, mask-overlap JSON, complete validation result ZIP, and GPU timing JSON, plus this CSV; record SHA-256.
3. Freeze the checkpoint, layer set, preprocessing, and eight selected configs before running the **official held-out test** once.
4. If time allows, repeat T4 timing over several independent benchmark runs and capture per-image predictions for paired bootstrap accuracy confidence intervals.
5. Report theoretical attention-edge sparsity and measured runtime separately; be explicit that prototype masking has computational overhead.

## Raw sources

- `gpu_latency_v4.json`: CUDA/T4 per-configuration latency medians.
- `Sparge_Full_Validation_v4.zip`: dense, unmasked, and eight sparse validation metrics on 736 images.
- `calibration_64_v4.json`: 64-image validation-only configuration selection.
- `mask_overlap_val.json`: direct mask comparisons from identical dense Q/K inputs.
