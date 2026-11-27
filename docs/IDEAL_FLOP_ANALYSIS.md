# Ideal block-skipping arithmetic estimate — November 27, 2026

**Scope:** Full validation (736 images); theoretical gross savings per image, not measured FLOPs, throughput or latency. No official test inference.

## Method

DeiT-Tiny: 197 tokens (196 patches + CLS), embedding width 192, 3 heads of width 64, 12 Transformer blocks, MLP width 768. The selected zero-based layers are 3, 6, 9. Count one multiply-accumulate as two FLOPs (usual matrix-product approximation; ignore boundary corrections for reductions).

The dense QK^T and AV products together cost `4*N*N*d = 29,805,312` FLOPs per layer and `12*4*N*N*d = 357,663,744` FLOPs across all layers. For measured sparsity s (fraction), gross ideal saved FLOPs are `3*s*4*N*N*d`. The relative reduction across attention products in all 12 layers is `3*s/12`.

Source check: `sparse_pet/attention.py` measures `(~keep).float().mean()` across the full B,H,N,N mask and weights batch means by image count. `sparse_pet/evaluate.py` averages selected-layer statistics. `sparse_pet/masking.py` keeps the CLS row and column. Thus the recorded sparsity already incorporates CLS connections and protected blocks; do not multiply by the patch-only fraction again.

An optional dense-model matrix-product denominator includes patch embedding, QKV/output projections, MLP linear maps and 37-class head:

`12*(4*N*N*d + 8*N*d*d + 4*N*d*768) + 2*196*(16*16*3)*d + 2*d*37 = 2,506,996,608` FLOPs.

This is a restricted matrix-product arithmetic estimate, not a full profiler count. It excludes biases, scaling, softmax, normalization, nonlinearities, residual additions and mask generation.

## Results

| Configuration | Realized validation sparsity (%) | Gross ideal saved MFLOPs/image | All-layer attention-product reduction (%) | Dense-model matrix-product reduction (%) |
|---|---:|---:|---:|---:|
| dense | 0.00 | 0.00 | 0.00 | 0.00 |
| unmasked | 0.00 | 0.00 | 0.00 | 0.00 |
| A_25 | 23.74 | 21.23 | 5.94 | 0.85 |
| B_25 | 25.68 | 22.96 | 6.42 | 0.92 |
| C_25 | 23.98 | 21.44 | 6.00 | 0.86 |
| D_25 | 25.55 | 22.85 | 6.39 | 0.91 |
| A_50 | 49.85 | 44.57 | 12.46 | 1.78 |
| B_50 | 52.62 | 47.05 | 13.16 | 1.88 |
| C_50 | 50.17 | 44.86 | 12.54 | 1.79 |
| D_50 | 51.76 | 46.28 | 12.94 | 1.85 |

## Interpretation and limitations

- A_50 yields 44.57 MFLOPs/image gross ideal savings, or 12.46% of all-layer attention matrix-product arithmetic and 1.78% of the restricted dense-model matrix-product denominator.
- A hypothetical implementation would have to build masks before full-resolution QK^T and skip masked products in both QK^T and AV. This is not what our code does. The actual wrapper computes both dense products; its mask-based arithmetic skipping savings are zero.
- The gross ideal estimates ignore block pooling, coarse QK^T, coarse softmax, sorting/top-k, similarity protection, mask expansion, memory traffic and sparse-kernel scheduling. Coarse QK^T alone requires approximately `3*2*49*49*192 = 2,765,952` FLOPs/image (excluding pooling/scaling). Therefore gross savings are not a prediction of net savings or latency.
- The observed Tesla T4 timings (FP32, batch 1, synthetic inputs, diagnostics off) remain the empirical performance evidence; every sparse variant was slower than native dense attention.
- Validation sparsity is measured, not assumed from the target; recompute with realized test sparsity after the team-approved official evaluation. No test-based recalibration.

## Report-ready paragraph

Using the measured full-validation sparsity, we estimate ideal gross arithmetic savings for a hypothetical block-skipping implementation. For DeiT-Tiny with 197 tokens and embedding width 192, the two attention matrix products require approximately 29.81 MFLOPs per layer. Because masks are applied at only three of twelve layers, A_50 (49.85% realized sparsity) would avoid 44.57 MFLOPs per image, corresponding to 12.46% of the attention matrix-product arithmetic across all layers, or 1.78% of our restricted dense-model matrix-product estimate. The latter excludes non-matrix operations and mask construction. These are theoretical gross estimates, not measured savings: our implementation computes dense QK^T and AV, and all sparse variants were slower than native dense attention on the T4 benchmark.

## Reproduction

Use the unrounded `diagnostics.realized_sparsity` values in `outputs/validation_v4/{configuration}.json` (dense has no diagnostic sparsity, defined as zero). For each configuration use:

```python
saved = 3 * 4 * 197**2 * 192 * s
attention_reduction_pct = 100 * saved / 357_663_744
matrix_product_reduction_pct = 100 * saved / 2_506_996_608
```

Inputs inspected: `DeiT_Sparse_Attention_GitHub_v4.zip` source and full-validation JSON records; `EXPERIMENT_FREEZE.md`. The estimate is a documentation-only addition; no evaluation parameters or model code changed.
