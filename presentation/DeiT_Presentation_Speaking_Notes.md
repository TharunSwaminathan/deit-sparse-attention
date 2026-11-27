# When to Trust Block Pooling? — speaking notes

**Five-slide implementation section | target 4 min 30 sec | draft for team integration**

Updated November 27 using the official held-out results. Validation diagnostics and synthetic GPU benchmark remain distinct. The explicit dated team approval record and dataset provenance confirmation are still open; coordinate the final handoff with Jiong.

## Slide 1 — Architecture (60 seconds)

Our implementation modifies attention during inference, using a single fine-tuned DeiT-Tiny checkpoint. Each 224-by-224 image produces a fourteen-by-fourteen patch grid plus a classification token. We group neighboring patches into two-by-two tiles, giving 49 spatial blocks, and average the query and key vectors within each block. The resulting coarse attention probabilities are used only to decide which block pairs to keep. Each retained pair expands to a four-by-four group of token-level edges, and every classification-token connection remains available. We then apply the mask to the original dense attention scores before softmax, so retained keys are renormalized and the original values are used. We apply this wrapper only at layers three, six and nine, using zero-based indices. Importantly, the prototype still computes dense score and value products; it is not a sparse GPU kernel.

## Slide 2 — Four masking rules (60 seconds)

We compare four strategies arranged as a two-by-two design. Variant A uses Top-p: it sorts the coarse probabilities and keeps the shortest prefix reaching the specified cumulative probability. Variant B adds similarity protection. We calculate the mean pairwise cosine similarity within each query or key block. If it is below the threshold, we preserve that entire coarse row or column. Variant C takes the union of Top-p and Top-k, guaranteeing a minimum number of retained key blocks in each row, apart from ranking ties. Variant D combines the hybrid rule with the same similarity protection. These choices let us examine both a minimum-count safeguard and a similarity safeguard. To compare methods fairly, we independently calibrate the parameters for approximately 25 and 50 percent sparsity in the selected layers. Because the parameters differ between variants, these comparisons evaluate complete strategies at similar budgets, rather than isolating a causal effect of a single safeguard.

## Slide 3 — Official test results (45 seconds)

We calibrated eight settings on 64 validation images and checked their budgets on all 736 validation images. On the official 3,669-image test set, dense and unmasked achieved 91.44 percent accuracy. Sparse accuracy ranged from 91.20 to 91.55 percent. A50 masked 49.91 percent of selected-layer edges and reached 91.55 percent accuracy. All sparse-versus-dense paired confidence intervals include zero, so we do not claim superiority or equivalence. Seven of eight budgets transferred within tolerance; B50 reached 53.134 percent. We report that deviation without test-based recalibration.

## Slide 4 — When to trust block pooling (60 seconds)

To investigate when pooling is reliable, we sampled three validation images per breed, 111 in total, and applied every mask to identical dense layer inputs without propagating sparse outputs. The heatmaps show Spearman correlations between relative output error and coarse entropy or query-block similarity. At layers three and six, 15 of 16 entropy correlations are positive; the strongest is 0.924 for C at the 25 percent target in layer three. Later-layer relationships are weaker or method dependent. Similarity alone is a weaker indicator. For all four protection comparisons in the lowest similarity quartile of layer three, error falls, but more connections are retained. For example, A50 to B50 reduces local error from about 16 to 5 percent while local sparsity drops from 59 to 12 percent. None of the protection pairs improves this quartile at layers six or nine. These are descriptive relationships, not causal effects, and motivate future layer-aware allocation.

## Slide 5 — Runtime and limitations (45 seconds)

Finally, our separate Tesla T4 benchmark used FP32, batch size one, ten warmups and thirty synchronized timed forwards. Dense inference took 4.74 milliseconds; sparse variants took 7.33 to 11.24 milliseconds, about 1.55 to 2.37 times slower. Although A50 could ideally save 44.63 million FLOPs per test image by skipping blocks, our prototype still performs dense score and value products. It adds pooling, ranking and masking overhead. The theoretical estimate excludes that overhead and is not measured acceleration. Our findings are limited to one checkpoint, one dataset and three selected layers. I’ll hand over to the team’s final evaluation discussion.

## Rehearsal reminders

- Keep layers zero-indexed.
- Use “percentage points” for accuracy differences.
- Never describe masked attention as GPU acceleration or statistical equivalence.
- Figure 06 (full local protection comparisons) is available in the repository for Q&A.
- Coordinate the test-result slide with Jiong to avoid duplicate coverage. If asked about provenance, explain that Jiong confirmed freeze use, but dated pre-run approval and independent dataset verification are still open. Never retune based on test data.
