# FS1000 Dynamic Warm-Start Design

## Goal

Initialize the first-token VideoMAE experiment from
`/home/v100/ZYQ/skating/fs800_result/checkpoint_best_0.872.pth` without
reusing the checkpoint's ResNet static branch or its static-aware fusion head.

## Loading Boundary

Load every checkpoint parameter except keys beginning with `static_proj.` or
`time_score_mlp.`. The resulting 34 dynamic parameters have exact name and
shape matches in the current model. Keep both excluded modules at their seeded
random initialization.

The checkpoint's `static_proj.weight` is `[128, 2048]`, while the first-token
model requires `[128, 768]`. The fusion head has a compatible shape but is also
excluded because it was trained against projected ResNet features.

## Training

Expose the source checkpoint through `--init-dynamic-checkpoint`. The manual
runner supplies the existing checkpoint by default and writes to a new run
directory. The existing two-stage schedule remains unchanged: epochs 0-29
train `static_proj + time_score_mlp`; epoch 30 onward trains all parameters.
The experiment remains deterministic with seed 2026.

## Verification

A unit test records the new model's randomly initialized static projection and
fusion head, loads a synthetic source state, and verifies that all dynamic
parameters change while both excluded modules remain byte-for-byte unchanged.
The loader also requires exact dynamic key and shape agreement.
