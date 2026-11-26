# When to Trust Block Pooling? — speaking notes

**Five-slide implementation section | target 4 min 30 sec | draft for team integration**

Results are validation-only and GPU benchmark results. Update the final handoff and held-out status after team approval and actual evaluation.

## Slide 1 — Architecture (60 seconds)

Our implementation modifies attention during inference, using a single fine-tuned DeiT-Tiny checkpoint. Each 224-by-224 image produces a fourteen-by-fourteen patch grid plus a classification token. We group neighboring patches into two-by-two tiles, giving 49 spatial blocks, and average the query and key vectors within each block. The resulting coarse attention probabilities are used only to decide which block pairs to keep. Each retained pair expands to a four-by-four group of token-level edges, and every classification-token connection remains available. We then apply the mask to the original dense attention scores before softmax, so retained keys are renormalized and the original values are used. We apply this wrapper only at layers three, six and nine, using zero-based indices. Importantly, the prototype still computes dense score and value products; it is not a sparse GPU kernel.

## Slide 2 — Four masking rules (60 seconds)

We compare four strategies arranged as a two-by-two design. Variant A uses Top-p: it sorts the coarse probabilities and keeps the shortest prefix reaching the specified cumulative probability. Variant B adds similarity protection. We calculate the mean pairwise cosine similarity within each query or key block. If it is below the threshold, we preserve that entire coarse row or column. Variant C takes the union of Top-p and Top-k, guaranteeing a minimum number of retained key blocks in each row, apart from ranking ties. Variant D combines the hybrid rule with the same similarity protection. These choices let us examine both a minimum-count safeguard and a similarity safeguard. To compare methods fairly, we independently calibrate the parameters for approximately 25 and 50 percent sparsity in the selected layers. Because the parameters differ between variants, these comparisons evaluate complete strategies at similar budgets, rather than isolating a causal effect of a single safeguard.

## Slide 3 — Calibration and validation (45 seconds)

We calibrated eight settings on 64 validation images and checked the budgets on all 736 validation images. All eight were within three percentage points of their respective targets. Both dense controls reached 93.75 percent accuracy, and the sparse configurations ranged from 93.61 to 94.02 percent. Those are differences of at most two correctly classified images relative to dense, not evidence of superiority or equivalence. At the lower budget, Top-p retained the most original attention mass and produced the lowest mean relative output error. At the higher budget, Top-p and the hybrid had almost identical output error. Since validation was also used for calibration and selection, we reserve generalization conclusions for the official test set.

## Slide 4 — When to trust block pooling (60 seconds)

To investigate when pooling is reliable, we sampled three validation images per breed, 111 in total, and applied every mask to identical dense layer inputs without propagating sparse outputs. The heatmaps show Spearman correlations between relative output error and coarse entropy or query-block similarity. At layers three and six, 15 of 16 entropy correlations are positive; the strongest is 0.924 for C at the 25 percent target in layer three. Later-layer relationships are weaker or method dependent. Similarity alone is a weaker indicator. For all four protection comparisons in the lowest similarity quartile of layer three, error falls, but more connections are retained. For example, A50 to B50 reduces local error from about 16 to 5 percent while local sparsity drops from 59 to 12 percent. None of the protection pairs improves this quartile at layers six or nine. These are descriptive relationships, not causal effects, and motivate future layer-aware allocation.

## Slide 5 — Runtime and limitations (45 seconds)

Finally, we measured median forward latency on a Tesla T4 using FP32, batch size one, ten warmups and thirty synchronized timed forwards. Native dense inference took 4.74 milliseconds, while sparse configurations took between 7.33 and 11.24 milliseconds. This is roughly 1.55 to 2.37 times slower because our implementation adds pooling, ranking and masking but still performs dense score and value products. Sparsity is therefore a measure of removed attention entries, not a measured reduction in total model FLOPs or a demonstrated speedup. Our findings are limited to one checkpoint, one dataset and three selected layers. The official held-out test evaluation is still pending approval; the results presented here are validation and benchmark results only. I'll hand over to the evaluation portion of the presentation.

## Rehearsal reminders

- Keep layers zero-indexed.
- Use “percentage points” for accuracy differences.
- Never describe masked attention as GPU acceleration or statistical equivalence.
- Figure 06 (full local protection comparisons) is available in the repository for Q&A.
- Coordinate the held-out results and bibliography with the team before the final presentation.
