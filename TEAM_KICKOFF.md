# Team kickoff — updated after professor's fine-tuning feedback

## Milestone 1 (now)

**Dev:** Set up official Oxford-IIIT Pet download, verify 37 labels and fixed
stratified train/val indices, train a **head-only** DeiT-Tiny pilot. Share
`outputs/dense_head/best.pt` in private team Drive and validation metrics.
Do not use test set to choose training strategy. Before full fine-tuning,
incorporate professor's Transformer fine-tuning recipe.

**Tharun:** Own `masking.py` and `attention.py`. Run synthetic tests, then
all-keep equivalence on a real randomly initialized/non-pretrained or loaded
timm model, then aggressive masks. Verify selected intermediate-layer masking affects model logits; a mask in only the final block may not affect CLS logits because the class-token row is preserved.

**Jiong:** Verify the metrics and timing harness, design the results table,
check that calibration uses validation only, and create script to plot
accuracy against **realized** sparsity. Cross-method attention diagnostics
must use identical inputs to each evaluated attention layer.

## Milestone 2 (next)

Agree on *one* dense checkpoint, lock the manifest and model/data config.
Implement A/B/C/D and pilot mask/calibration thresholds on a validation subset.
Save mask outputs plus p/k/threshold per variant; no Optuna necessary.

## Week 2 / Week 3

- Calibrate 25% and 50% sparsity on validation; report infeasible budgets.
- Run validation comparisons with a frozen checkpoint, common preprocess,
  common altered layers, consistent precision, and clear timing conditions.
- Freeze all choices, then evaluate the official test split only once per
  frozen configuration. Evaluate dense, unmasked, and all A-D configurations.
- Produce accuracy–sparsity plot, macro F1, diagnostic plot and latency table.
- Final report 6–8 pages plus GitHub and 12–15 minute demo.

## Research design caution

The professor questioned training the classifier plus last 3 layers for a
preset 6–10 epochs. This old line appears in the earlier internal plan and
is **superseded**. Attention layers are selected by preliminary validation analysis for *mask injection*, and the final chosen indices are shared across all four methods. Start
with a linear-probe baseline, then select full or partial fine-tuning based
on validation and the professor's new class material.

The code is an independently run starter, not a modification pushed to the
midterm GitHub repository. Repository access currently permits reading but
not pushing, so copy/merge these files into a team repository using Git.
