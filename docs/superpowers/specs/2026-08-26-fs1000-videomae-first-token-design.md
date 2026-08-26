# FS1000 VideoMAE First-Token Static Feature Ablation

## Goal

Measure whether retaining the first temporal token of the selected VideoMAE cliplet is better for FS1000 AQA than averaging all eight temporal tokens. This is a single-variable ablation of the completed mean-eight-token experiment.

## Fixed Experimental Conditions

- Use the existing FS1000 benchmark split: 651 training videos and 160 validation videos.
- Reuse the cached FS1000 VideoMAE features; do not decode videos or extract VideoMAE again.
- Reuse the frozen FineFS C1 ensemble with seeds 2026, 2027, and 2028.
- For MRU timestep `i`, keep the anchor `t = 2 * i` and candidate offsets `[0.0, 0.5, 1.0, 1.5]` seconds.
- Keep candidate construction and ensemble-confidence selection unchanged.
- Keep AST, TimeSformer, inverse features, MRU architecture, MSE objective, optimizer schedule, and the 200-epoch two-stage training protocol unchanged.
- Use training seed 2026 so the result is directly comparable with the completed mean-eight-token run.

## Only Experimental Change

After C1 selects a candidate, its first cliplet has shape `[8, 768]`. Store the first temporal token:

```python
static_vector = features[first_cliplet_index, 0]
```

The resulting vector remains 768-dimensional. The term "first token" means index 0 along the cached cliplet's temporal-token dimension; it does not introduce or select a separate CLS token. C1 still receives the same complete candidate features for confidence scoring.

## Isolation

- Branch: `experiment/fs1000-videomae-c1-first-token`
- Cache: `/media/v100/disk3t/skating/fs1000_static_videomae_c1_first_token`
- Run: `/media/v100/disk3t/skating/experiments/videomae_static_c1_first_token/run_seed2026`
- Cache metadata must report `first_temporal_token` rather than `mean_8_temporal_tokens`.

The completed mean-eight-token cache, checkpoints, logs, and branch remain unchanged.

## Verification

- A focused test must first fail under mean pooling and then pass when index-0 pooling is implemented.
- Generate 811 cache files with exact shape `[T_dynamic, 768]`, finite values, and full train/validation coverage.
- Record offset selection counts, incomplete candidate groups, and fallback count.
- Run 200 epochs with seed 2026 and independently reload both `best_spearman.pth` and `best_loss.pth`.

## Comparison

Compare against the completed mean-eight-token run:

- Best validation Spearman: `0.8472053656531183` at epoch 149.
- Best validation MSE: `74.6912654876709` at epoch 113.

Report the first-token best Spearman, best MSE, corresponding epochs, final-epoch behavior, and the deltas from these two reference values. This remains a one-seed ablation and must not be presented as a multi-seed estimate.
