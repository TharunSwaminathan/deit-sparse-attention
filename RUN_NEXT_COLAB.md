# v4 update: run THARUN_HANDOFF.md first for mechanism-activity aware calibration and identical-dense-input mask comparisons.

# Next Colab milestone: validation-only calibration (v3)

Your November 21 dense checkpoint, already trained on Oxford-IIIT Pet, is the baseline: **93.75% validation accuracy**, **93.79% macro F1**. Do **not** repeat training just to use this source update.

**What v3 changes from v2:** Calibration searches `top_k` values for hybrid masks instead of always using 4, searches similarity thresholds in a more plausible range, and records *how much* Top-k and similarity protection actually add. These additions help identify when A≈C or A≈B. They do **not** guarantee better results or a faster GPU kernel.

## A. Upload two ZIP files into a fresh Colab GPU runtime

Upload:

- `SpargeAttention_DeiT_Tiny_Project_v3.zip` (code)
- `Sparge_Project_Backup.zip` (model checkpoints, manifest, older results)

Then run in a Colab cell:

```python
!unzip -q -o /content/SpargeAttention_DeiT_Tiny_Project_v3.zip -d /content
%cd /content/sparge_pets_starter
!unzip -q -o /content/Sparge_Project_Backup.zip -d /content/sparge_pets_starter
!pip -q install -r requirements.txt
!python -m pytest -q tests
```

The backup already contains `outputs/dense_full_pilot/best.pt` and `outputs/split_manifest.json`. For a fresh runtime, the dataset itself may need downloading again; use `--download` below. Do not perform test-set evaluation in this step.

## B. Calibrate the four variants at 25% and 50% realized sparsity

After discussing candidate layer choices, a defensible **initial** set is `3,6,9` (zero-based), covering early, middle and later representations. Freeze this choice before any official test evaluation.

```python
!python -m sparse_pet.calibrate \
    --checkpoint outputs/dense_full_pilot/best.pt \
    --data-root /content/pets_data \
    --manifest outputs/split_manifest.json \
    --download \
    --layers 3,6,9 \
    --batch-size 8 \
    --images 64 \
    --targets 0.25 0.50 \
    --p-grid 0.70 0.82 0.90 0.95 0.98 0.99 \
    --top-k-grid 4 20 32 \
    --similarity-grid 0.50 0.70 0.85 \
    --output outputs/calibration_v3.json
```

This is an initial, modest validation-only search. It tries 96 configurations, so on Colab it will take noticeably longer than a single probe. It never trains model weights. If resources are limited, use `--images 32` first, then recalibrate/validate with more images before freezing anything.

## C. Inspect feasibility and actual mechanism activity

```python
import json
with open('outputs/calibration_v3.json') as f:
    calib = json.load(f)
print('Label  Target  Realized  In tolerance  p     k  sim   topk-added  protection-added')
for label, c in sorted(calib['selected'].items()):
    print(f"{label:5s}  {c['target_sparsity']:.2f}    {c['realized_sparsity']:.3f}"
          f"     {str(c['feasible_within_tolerance']):5s}   {c['p']:.2f}"
          f"  {c['top_k']:2d}  {str(c['similarity_threshold']):>4s}"
          f"     {c['topk_added_fraction']:.3f}         {c['protection_added_fraction']:.3f}")
for warning in calib['calibration_warnings']:
    print('WARNING:', warning)
```

Interpretation:

- `topk_added_fraction` = fraction of coarse block pairs **added beyond Top-p** by Top-k. If zero, hybrid does not change the mask relative to Top-p at the same `p`.
- `protection_added_fraction` = fraction of coarse pairs **added beyond the selection mask** by similarity protection. If zero, the protected variant is effectively unprotected.
- `feasible_within_tolerance` must be true for each label to call the 25% or 50% comparisons matched-budget (±3 pp). If false, expand the validation grid, report an infeasible target, or choose a shared realized budget. **Never silently force the requested sparsity**.

If protection is active but topk-added is zero (or vice versa), the method may still be useful as a negative result; document the mechanism activity honestly.

## D. Validate the calibrated settings, before touching the test set

```python
!python -m sparse_pet.evaluate \
    --checkpoint outputs/dense_full_pilot/best.pt \
    --data-root /content/pets_data --manifest outputs/split_manifest.json \
    --split val --frozen-config outputs/calibration_v3.json --label A_50 \
    --batch-size 8 --benchmark --output outputs/val_A_50_v3.json
```

Review the full validation result (all 736 images) and run the other seven labels only after confirming feasibility. The checkpoint is SHA-256-locked in the calibration file. The new source tree must be used for evaluation because the v2 calibration JSON schema differs in diagnostic fields. After all selections are frozen, official test evaluation can be performed once per configuration.

## E. Add the missing latest probe to your PC backup

The earlier backup ZIP omitted `outputs/layer_probe_full_p70.json`. If your earlier Colab session is still alive:

```python
from google.colab import files
files.download('outputs/layer_probe_full_p70.json')
```

Otherwise, its main values are visible in your exported Colab PDF; you can rerun that probe using the saved checkpoint and dataset manifest. Do not claim that the original file was archived unless you have downloaded it.

**Reproducibility note:** This pure PyTorch prototype still calculates dense attention scores before masking, so `realized_sparsity` describes **masked interactions**, not measured CUDA speedup. Include native dense, unmasked wrapper and masked timing with identical GPU, batch size, precision and warmup.
