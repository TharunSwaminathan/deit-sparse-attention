# Tharun's sparse-attention handoff (v4)

**Status:** local PyTorch tests passed (25). Real Oxford-IIIT Pet / pretrained DeiT-Tiny runs with this version still require a Colab GPU. The existing dense checkpoint (93.75% *validation* accuracy) is unchanged.

## Scope of my contribution

- Kept the existing 2x2 spatial pooling, Top-p, hybrid Top-k+Top-p, similarity protection, CLS preservation, and the timm attention wrapper.
- Added `sparse_pet/selection.py`: calibration picks configurations **within the ±3 percentage-point target tolerance first**, then prefers candidates with demonstrably active new mechanisms; infeasible and inactive results are flagged, not silently passed off as matched.
- Added `sparse_pet/compare_masks.py`: compares A/B/C/D masks on **identical dense Q/K activations** from validation examples and reports pairwise mask disagreement and Jaccard overlap.
- Added synthetic regression tests. New configuration defaults explore Top-k values of 12/24/36 and similarity thresholds 0.60/0.80; adjust via CLI based on validation evidence.

## Colab quick start

Upload `SpargeAttention_DeiT_Tiny_Project_v4.zip` and your existing `Sparge_Project_Backup.zip` into Colab, then:

```python
!unzip -q -o /content/SpargeAttention_DeiT_Tiny_Project_v4.zip -d /content
%cd /content/sparge_pets_starter
!unzip -q -o /content/Sparge_Project_Backup.zip -d /content/sparge_pets_starter
!pip -q install -r requirements.txt
!python -m pytest -q tests
```

First pilot: only 27 candidate configs with 24 validation images each:

```python
!python -m sparse_pet.calibrate \
    --checkpoint outputs/dense_full_pilot/best.pt \
    --data-root /content/pets_data --manifest outputs/split_manifest.json \
    --download --layers 3,6,9 --batch-size 8 --images 24 \
    --targets 0.25 0.50 \
    --p-grid 0.65 0.85 0.95 \
    --top-k-grid 16 32 \
    --similarity-grid 0.60 0.80 \
    --output outputs/calibration_pilot_v4.json
```

Inspect results without touching the test set:

```python
import json
r = json.load(open('outputs/calibration_pilot_v4.json'))
for label, c in sorted(r['selected'].items()):
    print(label, 'realized=', round(c['realized_sparsity'], 3),
          'matched=', c['feasible_within_tolerance'],
          'active=', c['mechanism_activity'],
          'p=', c['p'], 'k=', c['top_k'],
          'sim=', c['similarity_threshold'])
print('\n'.join(r['calibration_warnings']))
```

If no matched *active* configurations emerge, **expand the validation-only search**, not the test search. For a larger follow-up calibration use `--images 64 --p-grid 0.60 0.75 0.85 0.95 0.99 --top-k-grid 12 24 36 --similarity-grid 0.60 0.80`; this is **60 candidate evaluations**, so it may take appreciably longer on Colab T4. Save the chosen final file as `outputs/calibration_v4.json`.

After selecting and freezing the configuration, compare different rules on the same dense attention input:

```python
!python -m sparse_pet.compare_masks \
    --checkpoint outputs/dense_full_pilot/best.pt \
    --frozen-config outputs/calibration_v4.json \
    --data-root /content/pets_data --manifest outputs/split_manifest.json \
    --download --images 64 --batch-size 8 \
    --output outputs/mask_overlap_val.json
```

## Acceptance criteria / team integration

1. `pytest` passes on Colab; run the native vs all-kept smoke test again.
2. Every reported matched-budget configuration must have `feasible_within_tolerance=true`. Record inactive mechanisms even when no matched active candidate is feasible.
3. Mask overlap is measured on identical **dense** layer inputs; it is *not* equivalent to sparsity in actual multi-layer sparse inference. Jiong must run full validation on A–D separately.
4. Keep the official test split untouched until the checkpoint, layer set, calibration and reporting choices are frozen.
5. Do not claim GPU speedup from masked-edge counts: the prototype still computes the dense attention score matrix.
6. Share only source code and configs through GitHub. Distribute `.pt` checkpoints separately, with hashes and a clear provenance record.
