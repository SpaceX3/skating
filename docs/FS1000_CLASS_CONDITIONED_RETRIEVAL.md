# FS1000 Class-Conditioned VideoMAE Retrieval

## Experiment

For every FS1000 dynamic timestep:

1. Evaluate the four candidate 5-second windows at offsets `0, 0.5, 1.0, 1.5` with the frozen FineFS C1 ensemble.
2. Keep the most confident window and use the first token of its first 1-second cliplet as query `k` (`768D`).
3. Route `k` to the two classes with highest C1 probabilities. Class order is `background, jump, spin, sequence`.
4. Retrieve four cosine nearest FineFS training cliplets from each routed class.
5. Apply `softmax(similarity / 0.1)` within each class, then use the renormalized C1 probabilities across the two classes to obtain knowledge vector `m` (`768D`).
6. Save `concat(k, m)` as a `1536D` float16 static cache.

FineFS validation and test cliplets are not used in the bank. The MRU dynamic branch is initialized from `checkpoint_best_0.872.pth`; the `1536 -> 128` static projection and temporal score MLP remain randomly initialized.

No new package is required and the original `skating-action` environment is not modified.

## Manual Commands

```bash
cd /home/v100/.worktrees/skating-videomae-static
git switch experiment/fs1000-class-conditioned-retrieval
```

The commands below are intentionally expanded; no wrapper script is required.

Build the FineFS train-only four-class bank:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python build_finefs_class_bank.py \
  --manifest /media/v100/disk3t/finefs_pocr_classifier/experiments/e11_video_grain_split/cliplet_manifest.jsonl \
  --feature-root /media/v100/disk3t/finefs_pocr_classifier/features/videomae_base_1s_stride05 \
  --output-dir /media/v100/disk3t/skating/finefs_c1_class_bank_first_token
```

Precompute the FS1000 `concat(k, m)` cache:

```bash
CUDA_VISIBLE_DEVICES=GPU-f40ff723-535e-10df-74d9-4b38ebeac3c5 \
/home/v100/anaconda3/envs/skating-action-e10/bin/python precompute_class_conditioned_static.py \
  --dataset-root "/home/v100/ZYQ/FS1000 Dataset" \
  --feature-root /media/v100/disk3t/finefs_pocr_classifier/features/fs1000_videomae_base_1s_stride05 \
  --finefs-root /home/v100/ZYQ/finefs_pocr_classifier-e11-c1 \
  --c1-root /media/v100/disk3t/finefs_pocr_classifier/experiments/e11_c1_coarse_dense05/c1 \
  --bank-dir /media/v100/disk3t/skating/finefs_c1_class_bank_first_token \
  --output-dir /media/v100/disk3t/skating/fs1000_static_videomae_c1_class_retrieval \
  --device cuda:0 \
  --c1-batch-size 512 \
  --retrieval-batch-size 256 \
  --top-classes 2 \
  --top-k 4 \
  --temperature 0.1 \
  --probability-power 1.0
```

Train with the existing dynamic-branch warm start. The static projection and temporal score head are still randomly initialized:

```bash
CUDA_VISIBLE_DEVICES=GPU-f40ff723-535e-10df-74d9-4b38ebeac3c5 \
/home/v100/anaconda3/envs/skating-action/bin/python main.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --static-cache-dir /media/v100/disk3t/skating/fs1000_static_videomae_c1_class_retrieval \
  --static-cache-prefix static_videomae_c1_class_retrieval \
  --static-feature-dim 1536 \
  --init-dynamic-checkpoint /home/v100/ZYQ/skating/fs800_result/checkpoint_best_0.872.pth \
  --seed 2026 \
  --epochs 200 \
  --batch-size 16 \
  --num-workers 8 \
  --log-dir /media/v100/disk3t/skating/experiments/videomae_c1_class_retrieval/manual_seed2026_warmstart
```

Train from a completely random initialization. This is the requested no-checkpoint variant: leave out `--init-dynamic-checkpoint`, and every model parameter is initialized by `scoring_head`:

```bash
CUDA_VISIBLE_DEVICES=GPU-f40ff723-535e-10df-74d9-4b38ebeac3c5 \
/home/v100/anaconda3/envs/skating-action/bin/python main.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --static-cache-dir /media/v100/disk3t/skating/fs1000_static_videomae_c1_class_retrieval \
  --static-cache-prefix static_videomae_c1_class_retrieval \
  --static-feature-dim 1536 \
  --seed 2026 \
  --epochs 200 \
  --batch-size 16 \
  --num-workers 8 \
  --log-dir /media/v100/disk3t/skating/experiments/videomae_c1_class_retrieval/manual_seed2026_random
```

Evaluate both saved best checkpoints:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python eval.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --static-cache-dir /media/v100/disk3t/skating/fs1000_static_videomae_c1_class_retrieval \
  --static-cache-prefix static_videomae_c1_class_retrieval \
  --static-feature-dim 1536 \
  --checkpoint /media/v100/disk3t/skating/experiments/videomae_c1_class_retrieval/manual_seed2026_warmstart/best_spearman.pth

/home/v100/anaconda3/envs/skating-action/bin/python eval.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --static-cache-dir /media/v100/disk3t/skating/fs1000_static_videomae_c1_class_retrieval \
  --static-cache-prefix static_videomae_c1_class_retrieval \
  --static-feature-dim 1536 \
  --checkpoint /media/v100/disk3t/skating/experiments/videomae_c1_class_retrieval/manual_seed2026_warmstart/best_loss.pth
```

Expected generated locations:

- Bank: `/media/v100/disk3t/skating/finefs_c1_class_bank_first_token`
- FS1000 cache: `/media/v100/disk3t/skating/fs1000_static_videomae_c1_class_retrieval`
- Training log/checkpoints: `/media/v100/disk3t/skating/experiments/videomae_c1_class_retrieval/manual_seed2026`

The full FineFS bank is about 0.51 GB in float16. With the current 79,776 FS1000 timesteps, the 1536D float16 cache is about 0.25 GB.
