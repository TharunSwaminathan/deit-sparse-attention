# SparseAttention × DeiT-Tiny — Oxford-IIIT Pet (v3 starter)

**Research question:** At matched realized attention sparsity, do similarity
protection and hybrid Top-k+Top-p block selection help a pretrained DeiT-Tiny
maintain fine-grained classification accuracy?

For users who have already completed the fine-tuned baseline, **use `RUN_NEXT_COLAB.md`** to run v3 calibration directly without retraining.

This is a **self-contained experimental starter**, designed to be copied into a
new GitHub repository or merged into the team's midterm codebase. It does not
alter the existing CNN repository. There are **no pretrained weights, datasets,
or measured study results** in this archive.

## Research protocol

- Model: `timm` DeiT-Tiny **non-distilled** (`deit_tiny_patch16_224.fb_in1k`),
  224×224, 196 spatial patches plus one CLS token.
- Dataset: official Oxford-IIIT Pet, 37 breeds. Use its official trainval set
  for reproducible 80/20 stratified train/validation; preserve the official
  test set for once-after-freezing evaluation only.
- Dense transfer learning: start with a new 37-way head on frozen pretrained
  backbone; optional full-network Transformer fine-tuning should follow course
  guidance and validation results. The chosen blocks are **masking sites**,
  not a restriction on which layers may be fine-tuned.
- Four ablations: `A` Top-p; `B` Top-p + low-similarity protection; `C` hybrid
  Top-k/Top-p; `D` hybrid + protection.
- Spatial grouping: 2×2 patches -> 49 blocks; separate CLS token always kept
  in query and key interactions. Protection keeps the **entire block row or
  column** if within-block Q/K similarity is below threshold.
- Primary outcomes: top-1 accuracy, macro F1, realized masked fraction
  (including CLS pairs), retained mass, output error and latency.
- **Performance caveat:** the reference code constructs block masks but still
  performs a dense `Q @ K.T`; it *does not* provide a CUDA block-sparse kernel.
  Sparse masking may **slow down** inference. Report speed measurements honestly.

## Google Colab: first run

Unzip this archive into `/content/sparge_pets_starter`, then in Colab:

```bash
%cd /content/sparge_pets_starter
!pip install -r requirements.txt
!python -m pytest -q tests/

# Step 1: baseline (head-only pilot). Use a GPU runtime and official download.
!python -m sparse_pet.train --mode head --epochs 8 \
    --data-root /content/pets_data --manifest outputs/split_manifest.json \
    --batch-size 32 --download --output outputs/dense_head

# Step 2: validation-only dense check (no test evaluation yet).
!python -m sparse_pet.evaluate --checkpoint outputs/dense_head/best.pt \
    --data-root /content/pets_data --manifest outputs/split_manifest.json \
    --split val --variant dense --output outputs/val_dense.json

# Step 3: optional: whole-network fine-tuning of the best head-only checkpoint.
# Apply professor's Transformer fine-tuning advice before running this.
!python -m sparse_pet.train --mode full --epochs 5 --lr 1e-5 \
    --resume outputs/dense_head/best.pt --data-root /content/pets_data \
    --manifest outputs/split_manifest.json --output outputs/dense_full
```

**Choose only one final dense checkpoint for the entire masking study.** For
example, replace `CHECKPOINT` below with `outputs/dense_head/best.pt` or
`outputs/dense_full/best.pt` based on validation evidence. Do not choose a
checkpoint based on official test performance.

## Validate masks and run the experiments

```bash
# Optional quick sanity checks on validation data:
!python -m sparse_pet.evaluate --checkpoint CHECKPOINT \
    --data-root /content/pets_data --manifest outputs/split_manifest.json \
    --split val --variant A --layers 8,9,10 --p 0.7 --max-images 64 \
    --output outputs/pilot_A.json

# Check the unmasked wrapper against native dense DeiT:
!python -m sparse_pet.evaluate --checkpoint CHECKPOINT \
    --data-root /content/pets_data --manifest outputs/split_manifest.json \
    --split val --variant unmasked --layers 8,9,10 --max-images 64 --benchmark \
    --output outputs/unmasked_check.json

# Validation-only calibration for all four variants at target sparsities 25/50%:
!python -m sparse_pet.calibrate --checkpoint CHECKPOINT \
    --data-root /content/pets_data --manifest outputs/split_manifest.json \
    --layers 3,6,9 --images 64 --top-k-grid 4 20 32 \
    --similarity-grid 0.50 0.70 0.85 --output outputs/calibration.json

# Evaluate all four methods on VALIDATION with frozen candidate settings.
!python -m sparse_pet.evaluate --checkpoint CHECKPOINT \
    --data-root /content/pets_data --manifest outputs/split_manifest.json \
    --split val --frozen-config outputs/calibration.json --label A_25 \
    --output outputs/val_A_25.json --benchmark

# Only AFTER settings are frozen: one official TEST run per selected config.
!python -m sparse_pet.evaluate --checkpoint CHECKPOINT \
    --data-root /content/pets_data --manifest outputs/split_manifest.json \
    --split test --frozen-config outputs/calibration.json --label A_25 \
    --output outputs/test_A_25.json
```

Repeat frozen validation/test commands for `A_25`, `B_25`, `C_25`, `D_25`,
`A_50`, `B_50`, `C_50`, and `D_50`. The calibration now records `topk_added_fraction` and `protection_added_fraction` to catch ineffective rules. The `calibration.json` file reports whether
±3 percentage points of the target sparsity was feasible with this small grid.
Do not present infeasible pairs as matched-budget results; expand the
**validation** grid or report the nearest feasible shared budget.

Run a native dense timing baseline as well (`--variant dense --benchmark`).
For timing, diagnostic accumulation is disabled automatically. Use consistent
GPU, batch size, warmup, precision, and repetitions. Start with the code's
single-batch timing defaults and document any changes.

## File overview

- `sparse_pet/masking.py`: coarse 49×49 attention probabilities, Top-p, hybrid
  Top-k union, cosine-based low-similarity protection and CLS preservation.
- `sparse_pet/attention.py`: timm attention wrapper, mask before softmax,
  unmasked wrapper, and per-layer diagnostics.
- `sparse_pet/data.py`: dataset download and stratified split manifest.
- `sparse_pet/train.py`: dense baseline training; linear probe by default,
  optional full Transformer fine-tuning.
- `sparse_pet/calibrate.py`: small grid to approximately match realized sparsity
  based only on validation examples (no Optuna).
- `sparse_pet/evaluate.py`: full val/test classifier metrics and GPU timing.
- `tests/test_masks.py`: mask/CLS/protection/numerical tests on synthetic data.

## Limits and next engineering tasks

The v3 updates passed local synthetic tests but **have not yet been run
against the uploaded pretrained DeiT checkpoint on Oxford images in this offline
sandbox**. Version 2 was successfully exercised on the real checkpoint in Colab. On Colab, the first integration gate is whether the model's current
`timm` attention interface matches the wrapper. Run an all-kept (p=1)
model-level check against native DeiT logits (tolerance ~1e-4), then aggressive
mask check on selected blocks. Confirm image normalization with the exact
selected checkpoint.

Calibration uses a small validation subset for speed. Extend with locked full
validation evaluation before the official test. Diagnostic error uses dense
and sparse outputs for the **same input of each patched layer**. For strict
cross-method comparison using exactly the same dense activations for *every*
layer, add a cached dense activation diagnostic in Week 2; do not describe
current diagnostics as cross-method identical-input diagnostics.

Future: add layer-by-layer sparsity breakdowns and (optional) distribution
analysis by block cosine similarity and entropy; generate final plots and
bootstrap paired intervals. Account for mask construction FLOPs when discussing
ideal sparse-kernel savings; the present implementation does not skip dot
products and cannot claim real attention FLOP reductions.

## Reproducibility and licensing

Official dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/

Original paper: Parkhi et al., *Cats and Dogs*, CVPR 2012.
Oxford-IIIT Pet dataset page advertises CC BY-SA 4.0 licensing; individual
images may have separate copyright restrictions. Cite and link the publisher;
**do not commit images or checkpoints to a public repository** without proper
rights/license review.

Pretrained checkpoint: https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k

Papers: *SpargeAttention* (ICML 2025), *SpargeAttention2* (NeurIPS 2026).

Do not confuse this course-project prototype with exact replication of the
papers' optimized kernels or full diffusion-model training experiments.

## Layer-selection and dense-equivalence gate (new)

The submitted proposal does **not** preselect the final three blocks. The
`--layers 8,9,10` arguments above are **pilot examples only**, not a frozen
choice. Perform validation-only preliminary analysis and use the **same**
selected layer indices for A/B/C/D and both sparsity levels.

```bash
# No dataset, download or checkpoint required; randomized DeiT weights only.
!python -m sparse_pet.smoke_test --layers 8,9,10

# After Dev provides the dense checkpoint, examine layer effects on VALIDATION:
!python -m sparse_pet.probe_layers --checkpoint CHECKPOINT \
    --data-root /content/pets_data --manifest outputs/split_manifest.json \
    --layers 3,6,8,9,10,11 --images 32 --output outputs/layer_probe.json
```

A single masked **final** Transformer block may not change the classification
logits at all when all CLS attention rows are preserved. Intermediate layers
should be included when choosing the layer set. The smoke test compares native
attention with both the unmasked wrapper and the all-retained mask, then checks
that aggressive masking alters spatial-token attention and (when earlier layers
are selected) model outputs. It uses *random* model weights; this is a code
compatibility check, **not** a classification result.

On frozen evaluation, layer indices are loaded from the calibration JSON.
Passing different layer indices with `--layers` raises an error. Record the
reason for layer selection before test-set evaluation.

**Checkpoint locking:** `calibrate.py` stores the SHA-256 digest of the
checkpoint, and `evaluate.py` rejects a frozen calibration applied to
different model weights. Use identical input normalization, classes, and
validation split for all candidates.

## v4: Tharun's masking/calibration integration

See **`THARUN_HANDOFF.md`** for a short pilot protocol. New modules:
`selection.py` chooses budget-matched calibration candidates with active
masking mechanisms when possible, while `compare_masks.py` compares frozen
A–D masks on **identical dense layer inputs**. It provides mask-overlap
measurements but does not substitute for full-model evaluation. No extra
model training or custom CUDA kernel is required.
