# Validation-only block fidelity diagnostics

Run after the prediction-export patch. This is a mechanism study, not a new calibration, and does not use the official test split. It captures Q/K/V at the selected native dense attention layers, constructs each of the eight existing frozen masks on identical inputs, and compares masked versus full attention outputs before output projection. It never replaces attention in the model or changes model weights.

Default sample: three validation images per class, 111 images total, drawn without replacement with NumPy seed 42 from the saved trainval validation indices. The exact sample IDs are stored in the summary. The sample is exploratory and overlaps earlier validation experiments; it is not an independent test. Use the same subset for every configuration. `--per-class 1` is an optional 37-image functionality pilot only.

Per image, layer, head, and spatial query block (four query tokens), export query-block mean pairwise cosine similarity, mean key-block similarity for that head/image, coarse entropy (nats), retained dense attention mass, mean relative pre-projection output error, and realized sparsity. Exclude CLS query rows but keep CLS keys; this sparsity denominator differs from the full-model aggregate including CLS queries. Coarse entropy excludes CLS keys, as in the existing mask builder. Relative error is the mean of per-token L2 error divided by dense output norm (floor 1e-8). Key similarity is contextual, not a key-block-specific error measurement.

The summary gives descriptive Spearman correlations of query-block similarity and entropy with output error, and means/counts in layer-specific quartiles. Cutpoints are derived from the same dense input features and shared across the eight configurations. Ties can produce unequal or empty bins. Compare A/B and C/D at each target with actual local sparsity reported; p and k can differ, so these are not isolated causal estimates of protection. Blocks/heads within an image are dependent; do not interpret the correlations as independent-sample significance tests. This study does not measure errors propagated through the multi-layer sparse network, accuracy, or latency.

Example (PowerShell, from repository root, with existing variables):

```powershell
& $python -m sparse_pet.block_diagnostics `
  --checkpoint $checkpoint `
  --data-root $dataRoot `
  --manifest outputs/split_manifest.json `
  --frozen-config outputs/calibration_64_v4.json `
  --per-class 3 --batch-size 8 --workers 0 `
  --output-dir "$env:TEMP\deit-block-diagnostics"
```

Outputs: `block_diagnostics.json` (summary) and `query_blocks.csv.gz` (raw compressed CSV). Keep both for analysis and share the summary first. The script uses the existing SciPy dependency of scikit-learn. No dataset download is attempted. No official test loader is constructed. Three new synthetic unit tests check full-mask equivalence, shared features/protection, sampling and stratification. Full real-checkpoint run remains to be verified by the team.
