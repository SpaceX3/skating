# FS1000 Top-4 Score-Conditioned Cross-Attention

## Method

This experiment extends the existing FineFS Top-4 cross-attention cache with the
scores attached to each retrieved FineFS action interval. Each valid support stores
three values in this order:

```text
BV, GOE, score_of_pannel
```

The values are standardized using valid train-bank cliplets. Background support has
a zero descriptor and `score_valid_mask=0`. A cliplet that overlaps an annotated
action inherits that action interval's score. If `score_of_pannel` disagrees with
`BV + GOE` by more than `0.05`, the derived value is used; this fixes the known
`1639` typo in `annotation/898.json` as `16.39`.

For each of the two C1-routed classes, visual features alone produce the attention
keys. Score information is attached only to the values:

```text
q_i = Wq(query)
k_ci = Wk(support_ci)
v_ci = Wv([support_ci, normalized_score_ci, score_valid_mask_ci])
class_context_c = Attention(q, k_c, v_c)
context = sum_c C1_weight_c * class_context_c
```

The same attention weights compute the valid support-score mean, standard deviation,
and coverage. A small MLP projects those statistics to the residual feature space:

```text
z = LayerNorm(query + context_projection + score_statistics_projection)
```

The MRU dynamic branch and the final temporal scoring path are otherwise unchanged.

The float16 cache layout is:

```text
query 768
+ supports 2 x 4 x 768
+ normalized scores 2 x 4 x 3
+ score masks 2 x 4
+ C1 class weights 2
= 6946 values per timestep
```

## Manual Commands

Use the isolated experiment worktree:

```bash
cd /home/v100/.worktrees/skating-score-conditioned-top4
git branch --show-current
```

The expected branch is `experiment/fs1000-score-conditioned-top4`.

Build the scored FineFS train-only bank. This creates a new directory and does not
overwrite the previous feature-only bank:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python build_finefs_class_bank.py \
  --manifest /media/v100/disk3t/finefs_pocr_classifier/experiments/e11_video_grain_split/cliplet_manifest.jsonl \
  --feature-root /media/v100/disk3t/finefs_pocr_classifier/features/videomae_base_1s_stride05 \
  --annotation-dir /home/v100/ZYQ/FineFS/annotation \
  --output-dir /media/v100/disk3t/skating/finefs_c1_scored_bank_first_token
```

Precompute the FS1000 Top-4 cache on GPU 0:

```bash
/home/v100/anaconda3/envs/skating-action-e10/bin/python precompute_top4_cross_attention_static.py \
  --dataset-root "/home/v100/ZYQ/FS1000 Dataset" \
  --feature-root /media/v100/disk3t/finefs_pocr_classifier/features/fs1000_videomae_base_1s_stride05 \
  --finefs-root /home/v100/ZYQ/finefs_pocr_classifier-e11-c1 \
  --c1-root /media/v100/disk3t/finefs_pocr_classifier/experiments/e11_c1_coarse_dense05/c1 \
  --bank-dir /media/v100/disk3t/skating/finefs_c1_scored_bank_first_token \
  --output-dir /media/v100/disk3t/skating/fs1000_static_videomae_c1_top4_score_attention \
  --device cuda:0 \
  --c1-batch-size 512 \
  --retrieval-batch-size 256
```

Train using the existing MRU dynamic checkpoint. The score-aware cross-attention,
static projection, and temporal classifier are newly initialized. They train alone
for 30 epochs before the MRU parameters are unfrozen:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python main.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --static-cache-dir /media/v100/disk3t/skating/fs1000_static_videomae_c1_top4_score_attention \
  --static-cache-prefix static_videomae_c1_top4_score_attention \
  --static-feature-dim 6946 \
  --init-dynamic-checkpoint /home/v100/ZYQ/skating/fs800_result/checkpoint_best_0.872.pth \
  --freeze-backbone-epochs 30 \
  --seed 2026 \
  --epochs 200 \
  --batch-size 16 \
  --num-workers 8 \
  --log-dir /media/v100/disk3t/skating/experiments/videomae_c1_top4_score_attention/seed2026_warmstart
```

Train all parameters from random initialization by omitting the checkpoint and using
zero frozen epochs:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python main.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --static-cache-dir /media/v100/disk3t/skating/fs1000_static_videomae_c1_top4_score_attention \
  --static-cache-prefix static_videomae_c1_top4_score_attention \
  --static-feature-dim 6946 \
  --freeze-backbone-epochs 0 \
  --seed 2026 \
  --epochs 200 \
  --batch-size 16 \
  --num-workers 8 \
  --log-dir /media/v100/disk3t/skating/experiments/videomae_c1_top4_score_attention/seed2026_random
```

Evaluate the best warm-start checkpoint:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python eval.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --static-cache-dir /media/v100/disk3t/skating/fs1000_static_videomae_c1_top4_score_attention \
  --static-cache-prefix static_videomae_c1_top4_score_attention \
  --static-feature-dim 6946 \
  --checkpoint /media/v100/disk3t/skating/experiments/videomae_c1_top4_score_attention/seed2026_warmstart/best_spearman.pth
```

To use another GPU, change both `--device cuda:0` during precomputation and `--gpu 0`
during training or evaluation to the same device index. `CUDA_VISIBLE_DEVICES` is not
required.

## Focused Verification

```bash
/home/v100/anaconda3/envs/skating-action/bin/python -m unittest \
  tests.test_finefs_class_bank \
  tests.test_class_conditioned_retrieval \
  tests.test_top4_cross_attention
```
