# FS1000 Random-Class Top-4 Score Attention Ablation

## Purpose

This is a classifier-routing ablation of the C1 Top-2 score-conditioned Top-4
experiment. FS1000 does not provide compatible per-timestep coarse-class ground
truth, so this experiment uses one uniformly random coarse class per timestep.

Only the class route changes. C1 still selects the highest-confidence query cliplet
from offsets `[t, t+0.5, t+1.0, t+1.5]`; Top-4 cosine retrieval inside the selected
class, FineFS score descriptors, score masks, score-conditioned cross-attention,
MRU dynamic branch, optimizer, and loss remain unchanged.

The route is sampled once during precomputation using seed `2026` and is stored in
the generated cache. It is not re-sampled during training. A per-video seed sequence
uses the video ordinal, so rebuilding one video does not perturb other videos.

The four possible routes are sampled uniformly:

```text
background, jump, spin, sequence
```

Because only one class is routed, the cache dimension is:

```text
query 768
+ supports 1 x 4 x 768
+ normalized scores 1 x 4 x 3
+ score masks 1 x 4
+ class weight 1
= 3857 values per timestep
```

## Manual Commands

Use the isolated experiment branch:

```bash
cd /home/v100/.worktrees/skating-random-class-top4-score
git branch --show-current
```

Expected branch:

```text
experiment/fs1000-random-class-top4-score
```

The following commands are intentionally manual. They were not started by this
change.

### 1. Build The Random-Class Cache

This reuses the existing scored FineFS train-only bank. Build that bank first only
if `/media/v100/disk3t/skating/finefs_c1_scored_bank_first_token` is absent:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python build_finefs_class_bank.py \
  --manifest /media/v100/disk3t/finefs_pocr_classifier/experiments/e11_video_grain_split/cliplet_manifest.jsonl \
  --feature-root /media/v100/disk3t/finefs_pocr_classifier/features/videomae_base_1s_stride05 \
  --annotation-dir /home/v100/ZYQ/FineFS/annotation \
  --output-dir /media/v100/disk3t/skating/finefs_c1_scored_bank_first_token
```

Precompute the random-one-class cache:

```bash
/home/v100/anaconda3/envs/skating-action-e10/bin/python precompute_top4_cross_attention_static.py \
  --dataset-root "/home/v100/ZYQ/FS1000 Dataset" \
  --feature-root /media/v100/disk3t/finefs_pocr_classifier/features/fs1000_videomae_base_1s_stride05 \
  --finefs-root /home/v100/ZYQ/finefs_pocr_classifier-e11-c1 \
  --c1-root /media/v100/disk3t/finefs_pocr_classifier/experiments/e11_c1_coarse_dense05/c1 \
  --bank-dir /media/v100/disk3t/skating/finefs_c1_scored_bank_first_token \
  --output-dir /media/v100/disk3t/skating/fs1000_static_videomae_random1_top4_score_attention \
  --cache-prefix static_videomae_random1_top4_score_attention \
  --routing-mode random-one \
  --routing-seed 2026 \
  --device cuda:0 \
  --c1-batch-size 512 \
  --retrieval-batch-size 256
```

### 2. Warm-Start Training

This keeps the existing training schedule: the new score-aware static branch and
temporal head train for 30 epochs before the MRU parameters are unfrozen.

```bash
/home/v100/anaconda3/envs/skating-action/bin/python main.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --static-cache-dir /media/v100/disk3t/skating/fs1000_static_videomae_random1_top4_score_attention \
  --static-cache-prefix static_videomae_random1_top4_score_attention \
  --static-feature-dim 3857 \
  --top-classes 1 \
  --init-dynamic-checkpoint /home/v100/ZYQ/skating/fs800_result/checkpoint_best_0.872.pth \
  --freeze-backbone-epochs 30 \
  --seed 2026 \
  --epochs 200 \
  --batch-size 16 \
  --num-workers 8 \
  --log-dir /media/v100/disk3t/skating/experiments/videomae_random1_top4_score_attention/seed2026_warmstart
```

### 3. Random Initialization Training

Omit `--init-dynamic-checkpoint` and use zero frozen epochs to randomize all model
parameters:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python main.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --static-cache-dir /media/v100/disk3t/skating/fs1000_static_videomae_random1_top4_score_attention \
  --static-cache-prefix static_videomae_random1_top4_score_attention \
  --static-feature-dim 3857 \
  --top-classes 1 \
  --freeze-backbone-epochs 0 \
  --seed 2026 \
  --epochs 200 \
  --batch-size 16 \
  --num-workers 8 \
  --log-dir /media/v100/disk3t/skating/experiments/videomae_random1_top4_score_attention/seed2026_random
```

### 4. Evaluation

```bash
/home/v100/anaconda3/envs/skating-action/bin/python eval.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --static-cache-dir /media/v100/disk3t/skating/fs1000_static_videomae_random1_top4_score_attention \
  --static-cache-prefix static_videomae_random1_top4_score_attention \
  --static-feature-dim 3857 \
  --top-classes 1 \
  --checkpoint /media/v100/disk3t/skating/experiments/videomae_random1_top4_score_attention/seed2026_warmstart/best_spearman.pth
```

To use another GPU, change `cuda:0` to the desired device in precomputation and
change `--gpu 0` to the same index for training/evaluation. `CUDA_VISIBLE_DEVICES`
is not required.

## Focused Checks

```bash
/home/v100/anaconda3/envs/skating-action/bin/python -m unittest \
  tests.test_finefs_class_bank \
  tests.test_class_conditioned_retrieval \
  tests.test_random_class_routing \
  tests.test_top4_cross_attention
```

