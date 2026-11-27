# Related work and final results integration — November 27, 2026

Working material for the team report. Official test results are now populated from Jiong’s supplied evaluation; no additional test inference was performed to prepare this file. Reference labels are provisional and should be renumbered in the assembled report.

## Related work (draft for integration)

Vision Transformers treat an image as a sequence of patch embeddings [1]. DeiT develops a data-efficient approach to training image Transformers [2]. Our experiment uses a non-distilled, pretrained DeiT-Tiny model adapted to Oxford-IIIT Pet, a 37-breed classification dataset [3]. SpargeAttention motivates similarity-aware coarse attention selection and combines filtering with efficient sparse execution [4]. SpargeAttention2 studies the limitations of fixed-count and cumulative-mass selection and proposes a hybrid masking approach alongside sparse-attention fine-tuning [5]. Inspired by these methods, we compare four inference-time masking strategies at comparable realized sparsity using one fixed dense checkpoint. We study attention fidelity and classification rather than reproduce the diffusion-model training or optimized kernels of the original studies. FlashAttention illustrates the distinction between algorithmic masking and hardware-efficient execution [6]: our prototype continues to compute dense token-level attention products, so removed attention entries do not constitute measured acceleration.

## References (provisional numbering)

1. A. Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” ICLR, 2021. https://arxiv.org/abs/2010.11929
2. H. Touvron et al., “Training data-efficient image transformers & distillation through attention,” ICML, PMLR 139, pp. 10347–10357, 2021. https://proceedings.mlr.press/v139/touvron21a.html
3. O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, “Cats and Dogs,” CVPR, 2012. https://doi.org/10.1109/CVPR.2012.6248092 ; dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/
4. J. Zhang et al., “SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference,” ICML, PMLR 267, pp. 76397–76413, 2025. https://proceedings.mlr.press/v267/zhang25ch.html
5. J. Zhang et al., “SpargeAttention2: Trainable Sparse Attention via Hybrid Top-k+Top-p Masking and Distillation Fine-Tuning,” arXiv:2602.13515v2, 18 October 2026. https://arxiv.org/abs/2602.13515v2 . Cite the uploaded v2; confirm conference record and author list from the final proceedings before changing venue metadata. The earlier proposal lists NeurIPS 2026, but an official paper-specific conference record was not independently confirmed in this review.
6. T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,” NeurIPS, 2022. https://proceedings.neurips.cc/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract.html

## Verified held-out results table

Validation: 736 images. Official test: 3,669 images. Accuracy differences are candidate minus dense (pp).

| Configuration | Val sparsity (%) | Val accuracy (%) | Test sparsity (%) | Test accuracy (%) | Test macro-F1 | Test delta (pp) | 95% paired CI (pp) |
|---|---:|---:|---:|---:|---:|---:|---|
| dense | 0.00 | 93.75 | 0.00 | 91.44 | 0.9126 | — | — |
| unmasked | 0.00 | 93.75 | 0.00 | 91.44 | 0.9126 | 0.000 | [0.000, 0.000] |
| A_25 | 23.74 | 93.89 | 23.82 | 91.55 | 0.9137 | +0.109 | [0.000, +0.245] |
| B_25 | 25.68 | 93.89 | 25.79 | 91.55 | 0.9137 | +0.109 | [0.000, +0.245] |
| C_25 | 23.98 | 93.61 | 24.02 | 91.50 | 0.9131 | +0.055 | [-0.082, +0.218] |
| D_25 | 25.55 | 94.02 | 25.78 | 91.44 | 0.9126 | 0.000 | [-0.191, +0.191] |
| A_50 | 49.85 | 93.75 | 49.91 | 91.55 | 0.9137 | +0.109 | [-0.191, +0.409] |
| B_50 | 52.62 | 93.75 | 53.13* | 91.20 | 0.9101 | -0.245 | [-0.627, +0.136] |
| C_50 | 50.17 | 93.89 | 50.24 | 91.39 | 0.9121 | -0.055 | [-0.354, +0.245] |
| D_50 | 51.76 | 93.75 | 52.09 | 91.22 | 0.9103 | -0.218 | [-0.545, +0.136] |

*B_50 realized 53.133988% sparsity, 0.133988 pp above the 53% upper limit. All other sparse configurations met the ±3 pp tolerance. Dense/unmasked sparsity is zero by construction. CIs are marginal paired percentile intervals, not adjusted for multiplicity; all contain zero.

## Final report assembly

The 50 archive fingerprints, test IDs, labels, prediction-derived metrics and paired results were verified in the independent audit. The recorded source revision is `b1ebf0736004f8564c26803636467b451cf1ea16`. Jiong confirmed use of the pre-run freeze on November 27; the actual dated team approval record and independent dataset provenance check remain open. No test-based tuning. See `docs/OFFICIAL_TEST_RESULTS_REVIEW.md`.

Use the official test table and Jiong’s test plots in the main results section. Keep validation calibration, the 111-image diagnostic study, the 32-image demo and the synthetic T4 benchmark separate. Compare complete calibrated strategies, not isolated causal effects. Ideal block-skipping estimates are now documented in `docs/IDEAL_FLOP_ANALYSIS.md` (validation) and `docs/THARUN_REPORT_SECTIONS.md` (test); do not treat them as measured speedup. Reference numbering, conference metadata for SpargeAttention2, final page allocation and Dev’s training/data details must be coordinated at assembly.
