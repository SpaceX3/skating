# FS1000 VideoMAE Dynamic-Branch Ablation

## Setting

- Dynamic video input replaces TimeSformer with VideoMAE.
- MRU timestep `j` is anchored at `2j` seconds.
- It reads VideoMAE cliplets starting at `2j + [0,1,2,3,4]` seconds.
- Each cliplet keeps all eight `768D` tokens; the five cliplets are concatenated into `[40,768]`.
- One AST token is concatenated with the 40 VideoMAE tokens, so the MRU token-mixing input contains 41 tokens.
- Tail positions repeat the final available VideoMAE cliplet.
- Both ablations start all model parameters randomly and train them from epoch 0.
- `--use-static-branch` is the only difference between the two training commands.

The dynamic VideoMAE cache is approximately 4.57 GiB for 79,776 timesteps in float16.

## Checkout

```bash
cd /home/v100/.worktrees/skating-videomae-static
git switch experiment/fs1000-videomae-dynamic-ablation
```

## Precompute Dynamic VideoMAE

This command is CPU-only and does not use `--gpu`:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python precompute_videomae_dynamic.py \
  --dataset-root "/home/v100/ZYQ/FS1000 Dataset" \
  --feature-root /media/v100/disk3t/finefs_pocr_classifier/features/fs1000_videomae_base_1s_stride05 \
  --output-dir /media/v100/disk3t/skating/fs1000_dynamic_videomae_5x8
```

## A0: VideoMAE Dynamic Branch Only

Do not pass `--use-static-branch`:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python main.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --dynamic-video-cache-dir /media/v100/disk3t/skating/fs1000_dynamic_videomae_5x8 \
  --dynamic-video-cache-prefix dynamic_videomae_5x8 \
  --freeze-backbone-epochs 0 \
  --seed 2026 \
  --epochs 200 \
  --batch-size 16 \
  --num-workers 8 \
  --log-dir /media/v100/disk3t/skating/experiments/videomae_dynamic_ablation/dynamic_only_seed2026
```

## A1: VideoMAE Dynamic + Top-4 Cross-Attention Static Branch

If the 6914D Top-4 cache does not exist, build it first:

```bash
/home/v100/anaconda3/envs/skating-action-e10/bin/python precompute_top4_cross_attention_static.py \
  --dataset-root "/home/v100/ZYQ/FS1000 Dataset" \
  --feature-root /media/v100/disk3t/finefs_pocr_classifier/features/fs1000_videomae_base_1s_stride05 \
  --finefs-root /home/v100/ZYQ/finefs_pocr_classifier-e11-c1 \
  --c1-root /media/v100/disk3t/finefs_pocr_classifier/experiments/e11_c1_coarse_dense05/c1 \
  --bank-dir /media/v100/disk3t/skating/finefs_c1_class_bank_first_token \
  --output-dir /media/v100/disk3t/skating/fs1000_static_videomae_c1_top4_cross_attention \
  --device cuda:0 \
  --c1-batch-size 512 \
  --retrieval-batch-size 256
```

Train with the static branch by adding `--use-static-branch`:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python main.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --dynamic-video-cache-dir /media/v100/disk3t/skating/fs1000_dynamic_videomae_5x8 \
  --dynamic-video-cache-prefix dynamic_videomae_5x8 \
  --use-static-branch \
  --static-cache-dir /media/v100/disk3t/skating/fs1000_static_videomae_c1_top4_cross_attention \
  --static-cache-prefix static_videomae_c1_top4_cross_attention \
  --static-feature-dim 6914 \
  --freeze-backbone-epochs 0 \
  --seed 2026 \
  --epochs 200 \
  --batch-size 16 \
  --num-workers 8 \
  --log-dir /media/v100/disk3t/skating/experiments/videomae_dynamic_ablation/dynamic_top4_seed2026
```

Neither command passes `--init-dynamic-checkpoint`: the TimeSformer MRU checkpoint has an incompatible token-mixing shape and would invalidate this ablation.

## Evaluation

Dynamic-only checkpoint:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python eval.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --dynamic-video-cache-dir /media/v100/disk3t/skating/fs1000_dynamic_videomae_5x8 \
  --dynamic-video-cache-prefix dynamic_videomae_5x8 \
  --checkpoint /media/v100/disk3t/skating/experiments/videomae_dynamic_ablation/dynamic_only_seed2026/best_spearman.pth
```

Dynamic plus Top-4 checkpoint:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python eval.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --dynamic-video-cache-dir /media/v100/disk3t/skating/fs1000_dynamic_videomae_5x8 \
  --dynamic-video-cache-prefix dynamic_videomae_5x8 \
  --use-static-branch \
  --static-cache-dir /media/v100/disk3t/skating/fs1000_static_videomae_c1_top4_cross_attention \
  --static-cache-prefix static_videomae_c1_top4_cross_attention \
  --static-feature-dim 6914 \
  --checkpoint /media/v100/disk3t/skating/experiments/videomae_dynamic_ablation/dynamic_top4_seed2026/best_spearman.pth
```
