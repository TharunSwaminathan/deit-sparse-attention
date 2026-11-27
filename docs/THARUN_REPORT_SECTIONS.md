# When to Trust Block Pooling? — methodology, implementation and results

**Team report contribution | 27 November 2026 | Updated draft for team integration**

This contribution integrates the completed V4 validation experiments and official held-out results supplied by Jiong on 27 November. Numerical verification has passed; explicit dated pre-run team approval and dataset-release provenance remain to be confirmed. Section numbering, references and page allocation are for the team to finalize.

## Methodology

### Model, data and experimental controls

We study inference-time block-sparse attention in the non-distilled DeiT-Tiny model (`deit_tiny_patch16_224.fb_in1k` in timm), initialized with ImageNet-1k pretrained weights and adapted to the 37-category Oxford-IIIT Pet classification task. The official trainval partition was split into 2,944 training and 736 validation images using a stratified split (seed 42); the 3,669-image official test partition was reserved for final evaluation. Evaluation uses the model-resolved timm preprocessing transform with 224 × 224 input images. All configurations reuse a single dense fine-tuned checkpoint, selected by validation loss. No sparse retraining is performed. The native dense model and a manual unmasked attention wrapper serve as controls, separating the effect of replacing the attention implementation from that of masking. The same three zero-indexed Transformer blocks (3, 6 and 9) are used for every sparse configuration, selected after single-layer validation probes to cover network depth rather than claimed to be globally optimal. The saved probe (`layer_probe_p70.json`) used the earlier head-only checkpoint, 128 validation images, Top-p with p = 0.70, and candidate layers 3, 6, 8, 9, 10 and 11. This was a preliminary layer-selection exercise; final calibration and all reported full-validation, block-diagnostic and GPU results use the full fine-tuned checkpoint.

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

To compare rules at comparable attention sparsity, we use target removal fractions of 25% and 50% in the three modified layers. The final validation-only calibration searched p ∈ {0.65, 0.85, 0.87, 0.95, 0.97, 0.98}, k ∈ {12, 16, 32} for hybrid methods, and τ ∈ {0.45, 0.50, 0.55} for protected methods. The final search evaluated 96 candidate settings (6 A, 18 B, 18 C and 54 D). On a fixed 64-image validation subset (the first 64 in the ordered validation loader), candidate selection first requires proximity to the target within ±3 percentage points, then prefers active mechanisms (each relevant added rule restores at least 0.2% of coarse block pairs), and then minimizes absolute budget error, with calibration accuracy and p used as tie-breakers. All eight chosen configurations satisfied both budget feasibility and mechanism activity. Their budgets were subsequently checked on all 736 validation images. The calibration subset is contained in the full validation split; this is not independent test verification.

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

These validation-selected parameters were recorded in the freeze document before held-out evaluation; Jiong’s archived test configurations match them. They are independently calibrated by method; comparisons therefore characterize complete strategies at similar nominal budgets, not causal effects of adding one rule while holding all parameters fixed.

### Evaluation and diagnostics

End-to-end classification is measured by accuracy, macro-F1 across all 37 classes and cross-entropy. For the masked layers, realized sparsity is the fraction of removed token-level edges, including CLS connections in the denominator, averaged over the selected layers. Retained mass is the sum of original dense attention probabilities over retained keys, averaged over query tokens, heads and images. Relative output error is the mean of ||o_dense−o_masked||₂/max(||o_dense||₂,10⁻⁸) per query token and head; dense reference attention is evaluated on the same layer inputs as the masked attention. In the full-model evaluation these inputs can already reflect earlier masks. Pairwise mask disagreement is additionally measured on identical dense layer inputs to verify that the four rules generate genuinely different masks.

A separate validation-only diagnostic experiment sampled three images per breed (111 images, seed 42) and captured Q/K/V from the dense model at layers 3, 6 and 9. Each calibrated rule was then applied to these identical activations without propagating sparse outputs. For every image, head and patch query block, we averaged token-level error, retained mass and sparsity over its four query tokens; CLS keys remained included. We recorded query-block cosine similarity and coarse entropy H_i = −Σ_j P_ij log P_ij. Shared layer-specific quartiles and Spearman rank correlations were used to describe relationships with relative error. There were 16,317 observations per layer/configuration, or 391,608 configuration-specific records across 24 combinations, from only 111 images. These are correlated observations, so we do not interpret them as independent samples or report naive significance tests.

### Runtime and reproducibility

Latency was measured using synthetic device-resident FP32 inputs at batch size one, with diagnostics disabled. The Tesla T4 experiment used 10 warm-up forward passes and 30 measured repetitions, with CUDA synchronization before and after each timed forward pass; the median is reported. The CPU experiment used five warm-ups and 20 measured repetitions. Timings include mask construction and full-model forward execution, but exclude data loading, preprocessing and host-device transfer. Synthetic inputs have the same shape and sampling distribution; they were not explicitly held identical between all benchmark calls. Code, calibration JSON, split manifest and aggregate results are versioned. The trained weights are distributed separately and verified by SHA-256. The optional per-image export records stable official-split indices, labels, predicted classes and correctness for aligned comparison. The prediction export and subsequent paired-analysis and official-workflow utilities are covered by the expanded 56-test suite, which passed on Windows. A 32-image validation-only integration pilot verified unique IDs, matched labels and CSV/JSON accuracy for dense and A_50; all 32 predicted classes agreed. This pilot is not representative of the entire validation set. The official test launcher verifies frozen input fingerprints, prints the ten evaluation commands in dry-run mode, and requires an explicit approved revision and human-supplied approval reference for execution. The merged Windows implementation (`2915614fdcd4e219b7c039a96cdb8e073f48bc11`) passed all 56 tests and dry-run preflight. Jiong’s actual official test run used the integrated feature-branch revision `b1ebf0736004f8564c26803636467b451cf1ea16`, not the merge commit. The archived run records a clean checkout, the frozen input fingerprints, batch size 16, zero workers, FP32 and a Tesla T4 GPU, with timm 1.0.31, PyTorch 2.14.1+cu130 and bicubic resize to 248 followed by center crop to 224 and ImageNet normalization. The complete test execution and paired analysis took about seven minutes; this is not a per-image latency benchmark.

## Validation results and discussion (coordinate with results-section owner)

On 736 validation images, the native dense baseline and unmasked wrapper both achieved 93.75% accuracy. The eight sparse configurations achieved 93.61–94.02% accuracy while meeting their target budgets, corresponding to 689–692 correct predictions versus 690 for dense. These small differences do not establish superiority or statistical equivalence. At the 25% budget A retained the most dense attention mass (98.33%) and had the smallest mean relative output error (2.85%). At the 50% budget A and C had similar retained mass (88.39% and 88.52%) and relative error (15.59% and 15.64%). Protection did not consistently improve overall fidelity at these budgets. All pairwise mask comparisons at each target showed nonzero disagreement on identical dense inputs.

The block-level experiment indicates a layer-dependent relationship between entropy and fidelity: 15 of 16 entropy–error correlations in layers 3 and 6 were positive, reaching ρ = 0.924 for C_25 in layer 3; B_50 at layer 3 was an exception (ρ = −0.405). Layer 9 relationships were weaker or method dependent. Query similarity had negative correlations with error in 11 of 12 unprotected A/C combinations, but |ρ| did not exceed 0.200. In the lowest query-similarity quartile at layer 3, all four protected configurations reduced mean error relative to their corresponding unprotected configurations while retaining more connections locally; for example, A_50 versus B_50 reduced error from 16.17% to 5.33% while local sparsity fell from 58.64% to 11.52%. None of the four protection comparisons improved this quartile's error at layers 6 or 9. These results suggest that entropy and similarity may inform layer-dependent allocation, but do not demonstrate a superior adaptive policy. The separate diagnostic B_50 run realized 53.29% sparsity under its patch-query scope, outside the 47–53% nominal band; the full-validation result was 52.62%. We retain both measurements and do not recalibrate from the diagnostic run.

The GPU median latency was 4.739 ms for dense and 5.235 ms for the unmasked wrapper, compared with 7.327–11.244 ms for sparse variants (1.55–2.37 times the dense latency). The prototype computes dense token-level scores and dense attention-value products and incurs extra pooling, ranking and masking operations. It therefore demonstrates accuracy/fidelity trade-offs, not actual GPU acceleration.

## Official held-out test results

**Table 1.** Classification on the 3,669-image official test split using the eight frozen configurations and two controls. Accuracy differences are candidate minus dense, in percentage points. No test-based recalibration was performed in the supplied run.

| Configuration | Test sparsity (%) | Test accuracy (%) | Macro-F1 | Δ accuracy (pp) | 95% paired CI (pp) |
|---|---:|---:|---:|---:|---|
| dense | 0.00 | 91.44 | 0.9126 | — | — |
| unmasked | 0.00 | 91.44 | 0.9126 | 0.000 | [0.000, 0.000] |
| A_25 | 23.82 | 91.55 | 0.9137 | +0.109 | [0.000, +0.245] |
| B_25 | 25.79 | 91.55 | 0.9137 | +0.109 | [0.000, +0.245] |
| C_25 | 24.02 | 91.50 | 0.9131 | +0.055 | [-0.082, +0.218] |
| D_25 | 25.78 | 91.44 | 0.9126 | 0.000 | [-0.191, +0.191] |
| A_50 | 49.91 | 91.55 | 0.9137 | +0.109 | [-0.191, +0.409] |
| B_50 | 53.13* | 91.20 | 0.9101 | -0.245 | [-0.627, +0.136] |
| C_50 | 50.24 | 91.39 | 0.9121 | -0.055 | [-0.354, +0.245] |
| D_50 | 52.09 | 91.22 | 0.9103 | -0.218 | [-0.545, +0.136] |

*B_50 realized 53.133988% sparsity, 0.133988 pp above the 53% upper limit. All other sparse configurations met the ±3 pp tolerance. Dense/unmasked sparsity is zero by construction. CIs are marginal paired percentile intervals, not adjusted for multiplicity; all contain zero.

The native dense model achieved 91.44% accuracy (3,355/3,669); the manual unmasked control reproduced every predicted class. Sparse accuracy ranged from 91.20% to 91.55%, with changes of −0.245 to +0.109 pp relative to dense. A_50 achieved 49.91% realized sparsity and 91.55% accuracy (3,359 correct), with 40 predicted-label disagreements (98.91% agreement). The observed +0.109 pp change has a marginal 95% CI of [−0.191, +0.409] pp. None of the eight sparse-versus-dense intervals excludes zero. These results support small observed accuracy changes for this checkpoint, not statistical superiority or equivalence.

For uncertainty, the paired analysis resamples aligned test images with replacement (10,000 IID resamples, seed 42) and reports percentile intervals for candidate-minus-dense accuracy. These pointwise intervals are exploratory, not multiplicity-adjusted, and do not capture training-seed variability. The zero-width unmasked interval is a consequence of identical predictions, not proof of model equivalence in general. Independent checks of the supplied archive, all 36,690 class predictions, aggregate accuracy, macro-F1 and paired statistics passed (see `docs/OFFICIAL_TEST_RESULTS_REVIEW.md`). Loss and attention diagnostics cannot be reconstructed from predicted labels alone.

### Fidelity and interpretation

At the 50% target, A_50 and C_50 retained 88.39% and 88.53% of original dense attention mass, with relative output errors of 15.50% and 15.53%. B_50 and D_50 retained 79.81% and 84.64%, with errors of 24.60% and 20.04%. The protected strategies did not consistently improve aggregate accuracy or fidelity under their calibrated settings. Because realized budgets and parameter choices differ, this comparison cannot isolate the causal effect of protection. The validation-only block diagnostics suggest that the usefulness of entropy and similarity is layer-dependent; we did not repeat those diagnostic experiments on test images. The small B_50 budget deviation is reported as observed, not corrected using test information.

### Ideal arithmetic savings versus measured runtime

Counting one multiply-accumulate as two FLOPs, the dense QKᵀ and AV products require 4N²d = 29.81 MFLOPs per layer for N = 197 and embedding width d = 192. For measured sparsity s across three modified layers, hypothetical gross block-skipping savings are 3s × 4N²d. A_50's test sparsity implies 44.63 MFLOPs saved per image, or 12.48% of attention matrix-product arithmetic across all twelve layers and about 1.78% of the restricted dense-model matrix-product estimate (2,507.00 MFLOPs). CLS connections are already counted in the measured token-mask sparsity. The estimate excludes mask generation, non-matrix operations, memory traffic and sparse-kernel overhead. The actual prototype skips neither dense QKᵀ nor AV, and the separate synthetic Tesla T4 benchmark measured slower execution for all sparse methods (1.55–2.37× dense latency). This is not evidence of achieved FLOP reduction or acceleration.

## Limitations and conclusion

Our results are restricted to one architecture, one fine-tuned checkpoint, one dataset, three selected layers and two nominal sparsity budgets. Validation data informed checkpoint, layer and masking selection; the official test data were used to evaluate the recorded configurations, not to select new settings. Seven of eight configurations transferred within tolerance. The block-level correlations are exploratory, based on correlated observations from 111 validation images; separately calibrated parameters and local sparsity differences limit causal attribution. We find that coarse block pooling can support substantial removal of attention edges with small observed accuracy changes, but that aggregate fidelity and protection benefits depend on layer and strategy. A layer-aware allocation policy and an efficient sparse kernel are future work, not implemented contributions.

Jiong confirmed on 27 November that he used the pre-run experimental freeze. The run records the approval reference as `Team approval`, but the notebook does not identify the dated pre-execution approval or approving participants. The team still needs to establish the actual approval record and independently confirm the dataset release and contents. If explicit pre-run approval cannot be established, disclose this rather than backdate it. The audit verifies the delivered artifacts, not independent GPU inference or the original image bytes.

## Integration notes (remove from final report)

- Dev: verify training hyperparameters, preprocessing and official dataset provenance. Jiong: preserve the executed notebook and original `official_test_v4_001.zip`; integrate the raw outputs, figures and run provenance via his PR. Do not distribute the dataset or checkpoint in the public repository.
- Use Jiong’s `derived/test_accuracy_vs_sparsity` and `derived/test_accuracy_delta_ci` figures as the primary held-out plots; caption their zoomed axes and pointwise intervals. Keep validation calibration, block diagnostics and synthetic GPU benchmark distinct. See `docs/RELATED_WORK_AND_RESULTS_TEMPLATE.md` for the combined table and provisional references, `docs/IDEAL_FLOP_ANALYSIS.md` for the validation estimate, and `docs/OFFICIAL_TEST_RESULTS_REVIEW.md` for audit detail.
- Coordinate the final 6–8 page team report and presentation; this contribution is an integration draft, not the complete submission. Do not claim an implemented adaptive policy, statistical equivalence, or acceleration. Confirm references and page budget with the team.
- Frozen checkpoint SHA-256: `ca12f33e158e30c08279ff8be5bd5b2a69be75e688a01eee59195fc3f198cb06`. Test evaluation revision: `b1ebf0736004f8564c26803636467b451cf1ea16`. The recorded code and input fingerprints are preserved in the archive. The archived raw result files are authoritative; do not change their contents.
