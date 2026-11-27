# Related work and final results integration — November 27, 2026

Working material for the team report. Final test metrics are pending; no official test inference was performed to prepare this file. Reference labels are provisional and should be renumbered in the assembled report.

## Related work (draft for integration)

Vision Transformers treat an image as a sequence of patch embeddings [1]. DeiT develops a data-efficient approach to training image Transformers [2]. Our experiment uses a non-distilled, pretrained DeiT-Tiny model adapted to Oxford-IIIT Pet, a 37-breed classification dataset [3]. SpargeAttention motivates similarity-aware coarse attention selection and combines filtering with efficient sparse execution [4]. SpargeAttention2 studies the limitations of fixed-count and cumulative-mass selection and proposes a hybrid masking approach alongside sparse-attention fine-tuning [5]. Inspired by these methods, we compare four inference-time masking strategies at comparable realized sparsity using one fixed dense checkpoint. We study attention fidelity and classification rather than reproduce the diffusion-model training or optimized kernels of the original studies. FlashAttention illustrates the distinction between algorithmic masking and hardware-efficient execution [6]: our prototype continues to compute dense token-level attention products, so removed attention entries do not constitute measured acceleration.

## References (provisional numbering)

1. A. Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” ICLR, 2021. https://arxiv.org/abs/2010.11929
2. H. Touvron et al., “Training data-efficient image transformers & distillation through attention,” ICML, PMLR 139, pp. 10347–10357, 2021. https://proceedings.mlr.press/v139/touvron21a.html
3. O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, “Cats and Dogs,” CVPR, 2012. https://doi.org/10.1109/CVPR.2012.6248092 ; dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/
4. J. Zhang et al., “SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference,” ICML, PMLR 267, pp. 76397–76413, 2025. https://proceedings.mlr.press/v267/zhang25ch.html
5. J. Zhang et al., “SpargeAttention2: Trainable Sparse Attention via Hybrid Top-k+Top-p Masking and Distillation Fine-Tuning,” arXiv:2602.13515v2, 18 October 2026. https://arxiv.org/abs/2602.13515v2 . Cite the uploaded v2; confirm conference record and author list from the final proceedings before changing venue metadata. The earlier proposal lists NeurIPS 2026, but an official paper-specific conference record was not independently confirmed in this review.
6. T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,” NeurIPS, 2022. https://proceedings.neurips.cc/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract.html

## Held-out results table template

Validation values below come from `docs/Tharun_Final_Validation_and_T4_GPU_Results.csv` (736 images). The test columns are deliberately pending (3,669 official test images per configuration). Accuracy differences and confidence intervals are candidate minus dense in percentage points (pp), not relative percentages.

| Configuration | Val sparsity (%) | Val accuracy (%) | Test sparsity (%) | Test accuracy (%) | Test macro-F1 | Test delta (pp) | 95% paired CI (pp) |
|---|---:|---:|---:|---:|---:|---:|---|
| dense | 0.00 | 93.75 | Pending | Pending | Pending | — | — |
| unmasked | 0.00 | 93.75 | Pending | Pending | Pending | Pending | Pending |
| A_25 | 23.74 | 93.89 | Pending | Pending | Pending | Pending | Pending |
| B_25 | 25.68 | 93.89 | Pending | Pending | Pending | Pending | Pending |
| C_25 | 23.98 | 93.61 | Pending | Pending | Pending | Pending | Pending |
| D_25 | 25.55 | 94.02 | Pending | Pending | Pending | Pending | Pending |
| A_50 | 49.85 | 93.75 | Pending | Pending | Pending | Pending | Pending |
| B_50 | 52.62 | 93.75 | Pending | Pending | Pending | Pending | Pending |
| C_50 | 50.17 | 93.89 | Pending | Pending | Pending | Pending | Pending |
| D_50 | 51.76 | 93.75 | Pending | Pending | Pending | Pending | Pending |

## Acceptance checklist for Jiong's results

- Confirm team freeze approval, exact evaluated code revision, checkpoint/calibration/manifest fingerprints and dataset provenance. Merged main revision verified on Windows: `2915614fdcd4e219b7c039a96cdb8e073f48bc11`; 56 tests and dry run passed. This is not itself approval.
- Request ten aggregate JSONs, ten aligned prediction CSVs, `paired_analysis.json` and run provenance. Verify 3,669 unique expected test IDs per configuration, aligned labels, correct counts and aggregate metrics.
- Read test accuracy and macro-F1 from scores; use measured sparsity from diagnostics for sparse variants and structural zero for dense/unmasked. Report actual realized sparsity even outside tolerance; do not retune.
- Read delta and pointwise 95% percentile CI from paired analysis. Record wins/losses and prediction disagreements. No multiple-comparison adjustment or training-seed uncertainty; a CI containing zero does not prove equivalence.
- Keep separate the 736-image validation study, 111-image diagnostics, 32-image demo and synthetic T4 benchmark. Do not label any of these as test outcomes.
- Compare A/B, C/D, A/C and B/D at each budget descriptively; calibrated parameters differ, and comparisons do not identify causal effects of a rule. For inferential direct comparisons between sparse variants, predefine the analysis and distinguish it from the dense-referenced planned analysis.
- Replace pending cells only after verification. Update conclusions with supported results; do not claim a new adaptive policy, speedup or statistical equivalence.

## Final report assembly notes

Keep the main table and accuracy–sparsity plot centered on test results; put validation calibration detail and exploratory diagnostics in separate clearly labeled sections. Keep the T4 latency table as a separate benchmark. The revised proposal mentioned ideal block-skipping FLOP estimates; none have been reported in the current prototype results, so explicitly note this scope deviation rather than claim completed estimates. Reconcile training details and page budget with Dev and Jiong. The live demo is a functionality check, not an additional held-out experiment.
