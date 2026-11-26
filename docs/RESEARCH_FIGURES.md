# Research figures — validation evidence (November 26, 2026)

All six figures are available in `figures/` as 300 dpi PNG and vector PDF. They are derived from the V4 validation and benchmark results, not the official test set. No new inference was performed. Use PDF for the report and PNG for slides.

## Reproduce

From the repository root in a Python environment with NumPy:

```sh
python -m pip install matplotlib
python scripts/make_research_figures.py
```

Generated with Matplotlib 3.10.8. The script reads `outputs/validation_v4/*.json` (all ten configurations; 736 images each), `outputs/gpu_latency_v4.json`, `outputs/block_diagnostics.json`, and `outputs/calibration_64_v4.json`. It checks sample counts, evaluation splits, benchmark settings, layer indices, and matching recorded checkpoint hashes for calibration, GPU timing, and diagnostics. The original aggregate validation JSONs do not include checkpoint hashes. There is no need to download data or load weights to regenerate these plots.

## Figure captions and interpretation

1. **Block pooling and masking architecture.** Schematic based on `sparse_pet/masking.py` and `sparse_pet/attention.py`. The 196 patch tokens are grouped spatially into 49 blocks; mean Q/K vectors determine coarse scores and a retained-block mask. Hybrid variants union Top-k and Top-p. Protected variants also retain rows/columns with low within-block Q/K cosine similarity. The mask is expanded to the full token grid, preserving all CLS edges, and applied to dense scores before softmax. The diagram illustrates the mechanism, not a measured mask. Dense score and value products remain.
2. **Validation classification accuracy versus actual attention sparsity.** The horizontal dashed line is the native dense/manual unmasked validation accuracy (93.75%). Shading shows the target tolerance of ±3 percentage points. Each sparse marker represents a configuration on the full 736-image validation set. All eight fall within the tolerance. Accuracy ranges from 93.61% to 94.02%; the y-axis is intentionally zoomed. Small differences correspond to one or two correct images relative to dense, not evidence of superiority. No confidence intervals are available for this full validation run.
3. **Retained dense attention mass versus actual attention sparsity.** The retained mass is the probability assigned by dense attention to retained edges before masked softmax, averaged across selected layers. The 25% target group preserves approximately 95.0–98.3% of mass; the 50% group preserves approximately 80.0–88.5%. Protection does not consistently improve this metric at matched overall targets. Parameters differ across variants. The y-axis is zoomed.
4. **Tesla T4 median model forward latency.** FP32, batch size 1, 10 warmup and 30 measured forwards with CUDA synchronization; synthetic GPU-resident inputs of the same shape/distribution (not guaranteed identical values). The dense reference is 4.739 ms, manual unmasked is 5.235 ms, and masked configurations range from 7.327 to 11.244 ms. Timings include mask construction, but exclude data preprocessing and transfer; no sparse kernel acceleration is implemented. Only medians are available; no error bars or cross-hardware speedup claims.
5. **Spearman correlations of approximation error with entropy and query similarity.** 111 class-balanced validation images, three layers, eight configurations. Each layer/configuration has 16,317 query-block records (111 images × 3 heads × 49 query blocks). All masks are evaluated on identical dense layer inputs, without masked propagation. Entropy–error correlation is positive in 15/16 layer-3 and layer-6 combinations; layer 9 and B_50 at layer 3 show exceptions. For unprotected variants, query-similarity/error correlations are negative in 11/12 combinations, but weak. Correlations are descriptive; repeated blocks within images and across configurations are dependent.
6. **Local protection trade-offs in the lowest query-similarity quartile.** Compare paired rules A/B and C/D at each target in each layer. At layer 3 all four protected rules have lower mean error in this quartile, with lower local sparsity; at layers 6 and 9 none of the four pairs shows a lower mean error. The chart includes local sparsity beside error to avoid suggesting a free benefit. These are descriptive comparisons of different calibrated parameters, not isolated causal effects or locally budget-matched comparisons.

## Usage constraints

- A = Top-p; B = protected Top-p; C = hybrid; D = protected hybrid. Labels 25/50 denote nominal targets, not exact realized sparsity. Full-validation sparsity includes all attention entries within the three selected layers; it is not whole-model FLOP reduction.
- The 111-image block diagnostic experiment measures patch query blocks on dense inputs, and has a different metric scope from full-model validation. Do not merge its aggregate measurements into the full-validation accuracy plot.
- These figures support provisional conclusions only. Add held-out accuracy and paired confidence intervals separately after team approval and the actual official test run. Do not use the 32-image functionality pilot as inferential evidence.
