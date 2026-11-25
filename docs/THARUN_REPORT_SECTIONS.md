# When to Trust Block Pooling? — methodology and implementation draft

**Team report contribution | 25 November 2026 | Draft for integration**

This draft describes the implemented V4 methods and completed validation experiments. The official held-out test results and team-approved evaluation revision remain pending. Section numbers and bibliography numbering should be assigned when the team assembles the final report.

## Methodology

### Model, data and experimental controls

We study inference-time block-sparse attention in the non-distilled DeiT-Tiny model (`deit_tiny_patch16_224.fb_in1k` in timm), initialized with ImageNet-1k pretrained weights and adapted to the 37-category Oxford-IIIT Pet classification task. The official trainval partition was split into 2,944 training and 736 validation images using a stratified split (seed 42); the 3,669-image official test partition was reserved for final evaluation. Evaluation uses the model-resolved timm preprocessing transform with 224 × 224 input images. All configurations reuse a single dense fine-tuned checkpoint, selected by validation loss. No sparse retraining is performed. The native dense model and a manual unmasked attention wrapper serve as controls, separating the effect of replacing the attention implementation from that of masking. The same three zero-indexed Transformer blocks (3, 6 and 9) are used for every sparse configuration, selected after validation probes to cover network depth rather than claimed to be globally optimal.

### Block pooling and masking

For a given attention head, let Q, K and V denote the token-level query, key and value matrices, with head dimension d. A 224 × 224 image with 16 × 16 patches produces a 14 × 14 spatial patch grid and one classification token (CLS). We group spatially adjacent patches into nonoverlapping 2 × 2 tiles, yielding 49 blocks containing four tokens each. The mean query and key vectors within each block i are

$$\bar q_i=\frac{1}{4}\sum_{t\in i}q_t,\qquad \bar k_i=\frac{1}{4}\sum_{t\in i}k_t.$$

A coarse attention distribution is computed for each query block using mean-pooled representations:

$$P_{ij}=\operatorname{softmax}_j(\bar q_i^\top\bar k_j/\sqrt d).$$

The coarse probabilities guide selection only; the output uses original token-level Q, K and V, not pooled values. Four masking rules form a two-by-two comparison:

- **A (Top-p):** keep the shortest descending-probability prefix whose cumulative coarse probability is at least p, including the threshold-crossing block.
- **B (protected Top-p):** apply A, then retain all pairs incident to a query or key block whose mean within-block cosine similarity is below threshold τ.
- **C (hybrid):** take the union of Top-p and the k highest-probability key blocks per query row. As both rules rank the same scores, the union normally imposes a minimum retained-block count (apart from ties).
- **D (protected hybrid):** apply C followed by the same protection rule as B.

For a block of m = 4 nonzero query or key vectors, the similarity score is the mean off-diagonal pairwise cosine similarity, s = [1/(m(m−1))] Σ_(a≠b) cos(x_a,x_b). The protection condition s < τ restores an entire coarse query row or key column, rather than individual token edges. Each retained coarse pair expands to a 4 × 4 group of token-level attention edges. Every CLS query and key connection remains available, so the class token is never directly masked.

Let M be the resulting Boolean 197 × 197 token mask for each head. The attention output is calculated as

$$O_M=\operatorname{softmax}(QK^\top/\sqrt d+B_M)V,$$

where B_M is zero for retained positions and negative infinity for removed positions. Softmax renormalizes over retained keys. The original output projection, residual paths, feed-forward blocks and learned parameters are preserved. This is a masking prototype, not a fused sparse-attention kernel.

### Matched-budget calibration

To compare rules at comparable attention sparsity, we use target removal fractions of 25% and 50% in the three modified layers. The final validation-only calibration searched p ∈ {0.65, 0.85, 0.87, 0.95, 0.97, 0.98}, k ∈ {12, 16, 32} for hybrid methods, and τ ∈ {0.45, 0.50, 0.55} for protected methods. On a fixed 64-image validation subset, candidate selection first requires proximity to the target within ±3 percentage points, then prefers active mechanisms (each relevant added rule restores at least 0.2% of coarse block pairs), and then minimizes absolute budget error, with calibration accuracy and p used as tie-breakers. All eight chosen configurations satisfied both budget feasibility and mechanism activity. Their budgets were subsequently checked on all 736 validation images. The calibration subset is contained in the full validation split; this is not independent test verification.

| Configuration | p | k | τ | Full validation sparsity (%) |
|---|---:|---:|---:|---:|
| A_25 | 0.98 | — | — | 23.74 |
| B_25 | 0.97 | — | 0.50 | 25.68 |
| C_25 | 0.95 | 32 | — | 23.98 |
| D_25 | 0.65 | 32 | 0.55 | 25.55 |
| A_50 | 0.87 | — | — | 49.85 |
| B_50 | 0.65 | — | 0.55 | 52.62 |
| C_50 | 0.85 | 12 | — | 50.17 |
| D_50 | 0.65 | 16 | 0.50 | 51.76 |

The parameters are proposed for freezing before official test evaluation. They are independently calibrated by method; comparisons therefore characterize complete strategies at similar nominal budgets, not causal effects of adding one rule while holding all parameters fixed.

### Evaluation and diagnostics

End-to-end classification is measured by accuracy, macro-F1 across all 37 classes and cross-entropy. For the masked layers, realized sparsity is the fraction of removed token-level edges, including CLS connections in the denominator, averaged over the selected layers. Retained mass is the sum of original dense attention probabilities over retained keys, averaged over query tokens, heads and images. Relative output error is the mean of ||o_dense−o_masked||₂/max(||o_dense||₂,10⁻⁸) per query token and head; dense reference attention is evaluated on the same layer inputs as the masked attention. In the full-model evaluation these inputs can already reflect earlier masks. Pairwise mask disagreement is additionally measured on identical dense layer inputs to verify that the four rules generate genuinely different masks.

A separate validation-only diagnostic experiment sampled three images per breed (111 images, seed 42) and captured Q/K/V from the dense model at layers 3, 6 and 9. Each calibrated rule was then applied to these identical activations without propagating sparse outputs. For every image, head and patch query block, we averaged token-level error, retained mass and sparsity over its four query tokens; CLS keys remained included. We recorded query-block cosine similarity and coarse entropy H_i = −Σ_j P_ij log P_ij. Shared layer-specific quartiles and Spearman rank correlations were used to describe relationships with relative error. There were 16,317 observations per layer/configuration, or 391,608 configuration-specific records across 24 combinations, from only 111 images. These are correlated observations, so we do not interpret them as independent samples or report naive significance tests.

### Runtime and reproducibility

Latency was measured using synthetic device-resident FP32 inputs at batch size one, with diagnostics disabled. The Tesla T4 experiment used 10 warm-up forward passes and 30 measured repetitions, with CUDA synchronization before and after each timed forward pass; the median is reported. The CPU experiment used five warm-ups and 20 measured repetitions. Timings include mask construction and full-model forward execution, but exclude data loading, preprocessing and host-device transfer. Synthetic inputs have the same shape and sampling distribution; they were not explicitly held identical between all benchmark calls. Code, calibration JSON, split manifest and aggregate results are versioned. The trained weights are distributed separately and verified by SHA-256. The optional per-image export records stable official-split indices, labels, predicted classes and correctness for aligned comparison. It passed 38 local unit tests together with the diagnostic module, and a 32-image validation-only integration pilot verified unique IDs, matched labels and CSV/JSON accuracy for dense and A_50.

## Validation results and discussion (coordinate with results-section owner)

On 736 validation images, the native dense baseline and unmasked wrapper both achieved 93.75% accuracy. The eight sparse configurations achieved 93.61–94.02% accuracy while meeting their target budgets, corresponding to 689–692 correct predictions versus 690 for dense. These small differences do not establish superiority or statistical equivalence. At the 25% budget A retained the most dense attention mass (98.33%) and had the smallest mean relative output error (2.85%). At the 50% budget A and C had similar retained mass (88.39% and 88.52%) and relative error (15.59% and 15.64%). Protection did not consistently improve overall fidelity at these budgets. All pairwise mask comparisons at each target showed nonzero disagreement on identical dense inputs.

The block-level experiment indicates a layer-dependent relationship between entropy and fidelity: 15 of 16 entropy–error correlations in layers 3 and 6 were positive, reaching ρ = 0.924 for C_25 in layer 3; B_50 at layer 3 was an exception (ρ = −0.405). Layer 9 relationships were weaker or method dependent. Query similarity had negative correlations with error in 11 of 12 unprotected A/C combinations, but |ρ| did not exceed 0.200. In the lowest query-similarity quartile at layer 3, all four protected configurations reduced mean error relative to their corresponding unprotected configurations while retaining more connections locally; for example, A_50 versus B_50 reduced error from 16.17% to 5.33% while local sparsity fell from 58.64% to 11.52%. None of the four protection comparisons improved this quartile's error at layers 6 or 9. These results suggest that entropy and similarity may inform layer-dependent allocation, but do not demonstrate a superior adaptive policy. The separate diagnostic B_50 run realized 53.29% sparsity under its patch-query scope, outside the 47–53% nominal band; the full-validation result was 52.62%. We retain both measurements and do not recalibrate from the diagnostic run.

The GPU median latency was 4.739 ms for dense and 5.235 ms for the unmasked wrapper, compared with 7.327–11.244 ms for sparse variants (1.55–2.37 times the dense latency). The prototype computes dense token-level scores and dense attention-value products and incurs extra pooling, ranking and masking operations. It therefore demonstrates accuracy/fidelity trade-offs, not actual GPU acceleration.

## Limitations and final evaluation plan

The findings are limited to one model architecture, one fine-tuned checkpoint, one dataset, three selected attention layers, two nominal budgets and the measured hardware/protocol. Sparsity refers to attention entries in modified layers, not a measured reduction in total model FLOPs. Even an ideal sparse attention kernel would not remove the dense projections or MLP costs; no theoretical FLOP reduction or measured sparse-kernel speedup is claimed here. The 111-image diagnostics are exploratory and correlations do not isolate causal effects; differences in realized local sparsity and separately calibrated parameters complicate attribution. Validation results were obtained on data used for calibration and model/layer selection. The official test evaluation must use the approved checkpoint, split, parameters and revision without test-based retuning. Report actual test sparsity even when outside tolerance. Aligned per-image predictions will support paired image-level accuracy differences and confidence intervals; the final statistical protocol and test results remain to be supplied by the evaluation lead after team agreement. No official test outcomes are included in this draft.

## Integration notes (remove from final report)

- Dev: confirm exact training hyperparameters, preprocessing, model selection and bibliography entries for DeiT and Oxford-IIIT Pet.
- Jiong: insert held-out test results, paired uncertainty analysis, final figures and agreed test protocol; do not treat the 32-image pilot as a research result.
- Team: cite the original source of the sparse-attention inspiration and distinguish this prototype from its optimized implementation. Check references and page budget together. Do not imply a new entropy-adaptive method was implemented.
- Evidence in repository: `sparse_pet/masking.py`, `attention.py`, `selection.py`, `calibrate.py`, `evaluate.py`, `predictions.py`, `block_diagnostics.py`; `outputs/calibration_64_v4.json`, `outputs/split_manifest.json`, `outputs/validation_v4/`, `outputs/mask_overlap_val.json`, `outputs/gpu_latency_v4.json`, `outputs/block_diagnostics.json`; `docs/EXPERIMENT_FREEZE.md`, `docs/BLOCK_DIAGNOSTIC_RESULTS.md`, and `docs/Tharun_Final_Validation_and_T4_GPU_Results.csv`.
- Checkpoint SHA-256: `ca12f33e158e30c08279ff8be5bd5b2a69be75e688a01eee59195fc3f198cb06`. Diagnostic implementation and results pushed in commit `7f5ca7a`; PR #1 review/merge and final freeze approval are not established by this draft.
