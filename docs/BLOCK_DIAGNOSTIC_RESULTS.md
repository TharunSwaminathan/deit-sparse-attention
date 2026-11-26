# Block pooling reliability — validation diagnostics

## Scope and method

Exploratory validation-only experiment, November 25, 2026. The existing full fine-tuned DeiT-Tiny checkpoint (SHA-256 `ca12f33e158e30c08279ff8be5bd5b2a69be75e688a01eee59195fc3f198cb06`) was used without modifying weights or calibrated mask parameters. The CPU run sampled three validation images per breed (111 images, seed 42), using layers 3, 6 and 9 (zero-indexed). All eight configurations were evaluated on identical dense layer inputs; sparse outputs were not propagated. Patch query blocks were evaluated with CLS keys retained. The 49 spatial query blocks per head and three heads give 16,317 observations per layer/configuration, 48,951 distinct image/layer/head/query-block positions and 391,608 configuration-specific records. Quartile boundaries are shared across configurations within a layer. These records are correlated, not independent statistical samples.

## Overall diagnostic means

The table averages the three layer means equally (equal observation counts). Values are percentages of the respective metric. These are patch-query-block diagnostics, not end-to-end classification results or the original full-validation sparsity metric.

| Configuration | Query-block sparsity (%) | Retained mass (%) | Relative output error (%) |
|---|---:|---:|---:|
| A_25 | 24.02 | 98.32 | 2.85 |
| B_25 | 26.05 | 97.78 | 3.62 |
| C_25 | 24.24 | 96.93 | 5.20 |
| D_25 | 25.99 | 95.00 | 7.95 |
| A_50 | 50.61 | 88.30 | 15.63 |
| B_50 | 53.29 | 79.75 | 24.51 |
| C_50 | 50.90 | 88.50 | 15.62 |
| D_50 | 52.50 | 84.75 | 20.02 |

B_50 reaches 53.29% in this separate diagnostic protocol, slightly beyond the nominal 47–53% band. This does not contradict the original full-validation result (52.62%); the sample, input propagation and metric scope differ. Do not recalibrate from this exploratory result. A_25 has the lowest overall mean error at 25%; A_50 and C_50 are effectively tied descriptively at 50%. Protected variants have greater overall mean error than their unprotected counterparts at both nominal budgets.

## Rank correlations with relative output error

| Configuration | Entropy L3 | Entropy L6 | Entropy L9 | Query similarity L3 | Query similarity L6 | Query similarity L9 |
|---|---:|---:|---:|---:|---:|---:|
| A_25 | 0.702 | 0.489 | -0.039 | -0.008 | -0.151 | -0.106 |
| B_25 | 0.364 | 0.394 | -0.070 | 0.226 | -0.052 | -0.017 |
| C_25 | 0.924 | 0.838 | 0.044 | -0.090 | -0.197 | -0.175 |
| D_25 | 0.566 | 0.754 | 0.519 | 0.466 | 0.078 | 0.058 |
| A_50 | 0.572 | 0.484 | -0.157 | 0.062 | -0.165 | -0.174 |
| B_50 | -0.405 | 0.200 | -0.036 | 0.378 | 0.104 | 0.148 |
| C_50 | 0.885 | 0.697 | -0.145 | -0.094 | -0.200 | -0.186 |
| D_50 | 0.707 | 0.801 | 0.292 | 0.219 | -0.068 | -0.054 |

Entropy has positive rank correlation with error in 15 of 16 layer-3/6 combinations. B_50 at layer 3 is the exception (−0.405). Layer 9 shows weak, negative or method-dependent associations. Query similarity is negatively correlated with error in 11 of 12 unprotected A/C layer–configuration combinations, but the magnitudes are modest (absolute rho at most 0.200). Protected B/D configurations have mixed signs; protection changes the effective local sparsity and can reverse the relationship. These are descriptive associations, not causal or out-of-sample predictive tests.

## Lowest query-similarity quartile: protection comparison

Each cell gives unprotected → protected mean relative output error and local realized sparsity (percent). Pairs share the dense inputs and quartile boundaries, but differ in mask parameters and may have different actual sparsity.

| Layer | Pair | Error (%) | Local sparsity (%) |
|---|---|---:|---:|
| 3 | A_25 → B_25 | 3.07 → 2.29 | 28.17 → 19.51 |
| 3 | C_25 → D_25 | 4.91 → 1.09 | 30.07 → 5.07 |
| 3 | A_50 → B_50 | 16.17 → 5.33 | 58.64 → 11.52 |
| 3 | C_50 → D_50 | 16.58 → 9.08 | 59.56 → 31.03 |
| 6 | A_25 → B_25 | 3.18 → 3.82 | 18.52 → 20.17 |
| 6 | C_25 → D_25 | 6.16 → 7.39 | 24.12 → 21.41 |
| 6 | A_50 → B_50 | 16.68 → 21.70 | 46.40 → 42.94 |
| 6 | C_50 → D_50 | 18.21 → 22.44 | 48.87 → 50.23 |
| 9 | A_25 → B_25 | 2.61 → 3.18 | 7.98 → 9.50 |
| 9 | C_25 → D_25 | 6.39 → 9.27 | 17.16 → 21.18 |
| 9 | A_50 → B_50 | 15.58 → 19.51 | 34.88 → 36.31 |
| 9 | C_50 → D_50 | 17.73 → 25.40 | 38.36 → 47.13 |

At layer 3, all four protected variants have lower error in the lowest-similarity quartile, with lower local sparsity. At layers 6 and 9, none of the four protected variants has lower mean error in this quartile. These findings show localized trade-offs, not universal improvements or an isolated causal protection effect.

## Report-ready interpretation

Block-pooling reliability depends on layer and masking rule. Coarse-attention entropy is positively associated with local approximation error for most earlier and middle-layer configurations, while within-query-block cosine similarity exhibits weaker associations for unprotected masks. Similarity-based protection is associated with lower error for low-similarity query blocks in layer 3, but also lower local sparsity, and does not yield consistent improvements across depth. Overall, the protected configurations show higher mean local approximation error than their unprotected counterparts at the corresponding nominal budgets. These observations motivate layer-aware, entropy-sensitive sparsity allocation as future work; they do not establish that a new adaptive masking rule improves performance.

## Limitations and next steps

Only 111 validation images, one checkpoint, three layers and eight already-calibrated configurations were studied. Dense-input local error is not the same as end-to-end error or classification accuracy. The comparisons do not control actual local sparsity or isolate protection from differing p/k settings. The summary contains means and rank correlations, not image-clustered confidence intervals. Keep `query_blocks.csv.gz` outside Git for reproducibility and image-level resampling if required. Do not use the held-out test set to tune parameters; obtain team approval for the proposed freeze first.

Source: `outputs/block_diagnostics.json` (copy of the uploaded diagnostic summary).
