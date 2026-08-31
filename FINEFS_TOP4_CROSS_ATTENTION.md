# FineFS Top-4 Cross-Attention

This is the FineFS adaptation of `experiment/fs1000-top4-cross-attention`.
It uses the E11 video-level split and regresses each routine's
`total_element_score` (TES).

## Data contract

- Split: `/media/v100/disk3t/finefs_pocr_classifier/experiments/e11_video_grain_split/video_split.json`
- Annotation: `/home/v100/ZYQ/FineFS/annotation/{video_id}.json`
- AST: `/home/v100/ZYQ/finefs_av_feature_extractor/features/ast_feature_finefs/{video_id}.npy`, shape `[T,768]`
- Timesformer: `/home/v100/ZYQ/finefs_av_feature_extractor/features/Timesformer_output_feature_finefs/{video_id}.npy`, shape `[T,15,768]`
- VideoMAE query features: `/media/v100/disk3t/finefs_pocr_classifier/features/videomae_base_1s_stride05`

AST and Timesformer are already aligned in time, so this experiment does not
resample or average them. The existing FineFS C1 ensemble chooses one query
cliplet from offsets `0, 0.5, 1.0, 1.5` for each 2-second dynamic timestep.
The query is routed to its top-2 coarse classes, and cosine retrieval keeps
four train-split FineFS support vectors per routed class. The static cache is
stored as:

```text
query (768) + supports (2 x 4 x 768) + C1 class weights (2) = 6914 dimensions
```

The model is the existing MRU bidirectional dynamic branch plus the existing
4-head Cross-Attention module and a 128D static projection. With the warm-start
option, only the dynamic branch is loaded from the FS800 checkpoint; the
Cross-Attention, static projection, and temporal classifier are initialized
fresh. The first stage trains those new components for 30 epochs, then stage 2
unfreezes the complete model.

## Manual commands

Run the commands on the server after connecting as `v100`.

### 1. Select the experiment worktree

```bash
cd /home/v100/.worktrees/skating-finefs-top4-cross-attention
```

### 2. Build or refresh the train-only FineFS bank

The bank already exists at the default path. Run this only when rebuilding it:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python build_finefs_class_bank.py \
  --manifest /media/v100/disk3t/finefs_pocr_classifier/experiments/e11_video_grain_split/cliplet_manifest.jsonl \
  --feature-root /media/v100/disk3t/finefs_pocr_classifier/features/videomae_base_1s_stride05 \
  --output-dir /media/v100/disk3t/skating/finefs_c1_class_bank_first_token
```

### 3. Precompute FineFS raw Top-4 caches

This is the only precomputation specific to this adaptation. It uses the
existing C1 checkpoints and the existing VideoMAE features.

```bash
mkdir -p /media/v100/disk3t/skating/finefs_static_videomae_c1_top4_cross_attention

/home/v100/anaconda3/envs/skating-action/bin/python -u \
  precompute_finefs_top4_cross_attention_static.py \
  --split-json /media/v100/disk3t/finefs_pocr_classifier/experiments/e11_video_grain_split/video_split.json \
  --audio-feature-dir /home/v100/ZYQ/finefs_av_feature_extractor/features/ast_feature_finefs \
  --video-feature-dir /home/v100/ZYQ/finefs_av_feature_extractor/features/Timesformer_output_feature_finefs \
  --feature-root /media/v100/disk3t/finefs_pocr_classifier/features/videomae_base_1s_stride05 \
  --finefs-root /home/v100/ZYQ/finefs_pocr_classifier-e11-c1 \
  --c1-root /media/v100/disk3t/finefs_pocr_classifier/experiments/e11_c1_coarse_dense05/c1 \
  --bank-dir /media/v100/disk3t/skating/finefs_c1_class_bank_first_token \
  --output-dir /media/v100/disk3t/skating/finefs_static_videomae_c1_top4_cross_attention \
  --device cuda:0 \
  --c1-batch-size 512 \
  --retrieval-batch-size 256 \
  2>&1 | tee /media/v100/disk3t/skating/finefs_static_videomae_c1_top4_cross_attention/precompute.log
```

### 4. Train with the FS800 dynamic warm start

```bash
mkdir -p /media/v100/disk3t/skating/experiments/finefs_top4_cross_attention/seed2026_warmstart

/home/v100/anaconda3/envs/skating-action/bin/python -u main_finefs_top4.py \
  --mode train \
  --gpu 0 \
  --split-json /media/v100/disk3t/finefs_pocr_classifier/experiments/e11_video_grain_split/video_split.json \
  --annotation-dir /home/v100/ZYQ/FineFS/annotation \
  --audio-feature-dir /home/v100/ZYQ/finefs_av_feature_extractor/features/ast_feature_finefs \
  --video-feature-dir /home/v100/ZYQ/finefs_av_feature_extractor/features/Timesformer_output_feature_finefs \
  --static-cache-dir /media/v100/disk3t/skating/finefs_static_videomae_c1_top4_cross_attention \
  --static-cache-prefix static_videomae_c1_top4_cross_attention \
  --static-feature-dim 6914 \
  --init-dynamic-checkpoint /home/v100/ZYQ/skating/fs800_result/checkpoint_best_0.872.pth \
  --freeze-backbone-epochs 30 \
  --seed 2026 \
  --epochs 200 \
  --batch-size 16 \
  --num-workers 8 \
  --log-dir /media/v100/disk3t/skating/experiments/finefs_top4_cross_attention/seed2026_warmstart
```

The script writes an epoch-separated log plus `best_loss.pth`,
`best_spearman.pth`, `last.pth`, `best_metrics.json`, and
`resolved_config.json` in `--log-dir`.

### 5. Optional random-initialization control

Omit `--init-dynamic-checkpoint` and set `--freeze-backbone-epochs 0` so all
parameters start random and train jointly:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python -u main_finefs_top4.py \
  --mode train --gpu 0 \
  --split-json /media/v100/disk3t/finefs_pocr_classifier/experiments/e11_video_grain_split/video_split.json \
  --annotation-dir /home/v100/ZYQ/FineFS/annotation \
  --audio-feature-dir /home/v100/ZYQ/finefs_av_feature_extractor/features/ast_feature_finefs \
  --video-feature-dir /home/v100/ZYQ/finefs_av_feature_extractor/features/Timesformer_output_feature_finefs \
  --static-cache-dir /media/v100/disk3t/skating/finefs_static_videomae_c1_top4_cross_attention \
  --static-feature-dim 6914 --freeze-backbone-epochs 0 \
  --seed 2026 --epochs 200 --batch-size 16 --num-workers 8 \
  --log-dir /media/v100/disk3t/skating/experiments/finefs_top4_cross_attention/seed2026_random
```

### 6. Test a checkpoint

```bash
/home/v100/anaconda3/envs/skating-action/bin/python -u main_finefs_top4.py \
  --mode test --gpu 0 \
  --split-json /media/v100/disk3t/finefs_pocr_classifier/experiments/e11_video_grain_split/video_split.json \
  --annotation-dir /home/v100/ZYQ/FineFS/annotation \
  --audio-feature-dir /home/v100/ZYQ/finefs_av_feature_extractor/features/ast_feature_finefs \
  --video-feature-dir /home/v100/ZYQ/finefs_av_feature_extractor/features/Timesformer_output_feature_finefs \
  --static-cache-dir /media/v100/disk3t/skating/finefs_static_videomae_c1_top4_cross_attention \
  --static-feature-dim 6914 \
  --checkpoint /media/v100/disk3t/skating/experiments/finefs_top4_cross_attention/seed2026_warmstart/best_spearman.pth
```

The test command reports TES MSE, Spearman correlation, and Pearson
correlation on the E11 test split.
