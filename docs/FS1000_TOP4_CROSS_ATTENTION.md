# FS1000 Top-4 Cross-Attention Experiment

## Method

For each FS1000 dynamic timestep, the frozen FineFS C1 ensemble first selects the highest-confidence window among offsets `0, 0.5, 1.0, 1.5`. Its first cliplet token is query `k` (`768D`). C1 routes the query to its top-2 coarse classes. Cosine retrieval keeps four raw FineFS train-split support vectors for each routed class.

The float16 cache layout is:

```text
k (768) + top2 supports (2 x 4 x 768) + normalized C1 top2 weights (2)
= 6914 dimensions per timestep
```

The trainable fusion uses a shared 4-head Cross-Attention in a 128-dimensional attention space:

```text
q = Wq(k)
class_context_c = CrossAttention(q, Wk(B_c), Wv(B_c)), B_c = {b1,b2,b3,b4}
context = sum_c p_c * class_context_c
z = LayerNorm(k + Wo(context))
```

`z` is projected from `768D` to `128D` by the existing static projection and concatenated with the MRU dynamic representation. The gated baseline cache cannot be reused because it contains only the already-aggregated `m`; the existing FineFS four-class bank can be reused.

Expected Top-4 cache size for 79,776 timesteps is approximately 1.03 GiB.

## Manual Commands

Checkout the experiment branch:

```bash
cd /home/v100/.worktrees/skating-videomae-static
git switch experiment/fs1000-top4-cross-attention
```

Build the FineFS train-only bank only if it does not already exist:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python build_finefs_class_bank.py \
  --manifest /media/v100/disk3t/finefs_pocr_classifier/experiments/e11_video_grain_split/cliplet_manifest.jsonl \
  --feature-root /media/v100/disk3t/finefs_pocr_classifier/features/videomae_base_1s_stride05 \
  --output-dir /media/v100/disk3t/skating/finefs_c1_class_bank_first_token
```

Precompute the new raw Top-4 cache on GPU 0:

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

Train with the existing MRU dynamic checkpoint. Cross-Attention, static projection, and temporal classifier are newly initialized and trained for 30 epochs before unfreezing the MRU:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python main.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --static-cache-dir /media/v100/disk3t/skating/fs1000_static_videomae_c1_top4_cross_attention \
  --static-cache-prefix static_videomae_c1_top4_cross_attention \
  --static-feature-dim 6914 \
  --init-dynamic-checkpoint /home/v100/ZYQ/skating/fs800_result/checkpoint_best_0.872.pth \
  --freeze-backbone-epochs 30 \
  --seed 2026 \
  --epochs 200 \
  --batch-size 16 \
  --num-workers 8 \
  --log-dir /media/v100/disk3t/skating/experiments/videomae_c1_top4_cross_attention/seed2026_warmstart
```

Train all parameters from random initialization. Omitting `--init-dynamic-checkpoint` and setting the freeze duration to zero are both required:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python main.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --static-cache-dir /media/v100/disk3t/skating/fs1000_static_videomae_c1_top4_cross_attention \
  --static-cache-prefix static_videomae_c1_top4_cross_attention \
  --static-feature-dim 6914 \
  --freeze-backbone-epochs 0 \
  --seed 2026 \
  --epochs 200 \
  --batch-size 16 \
  --num-workers 8 \
  --log-dir /media/v100/disk3t/skating/experiments/videomae_c1_top4_cross_attention/seed2026_random
```

Evaluate a trained checkpoint on GPU 0:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python eval.py \
  --gpu 0 \
  --root-path "/home/v100/ZYQ/FS1000 Dataset" \
  --static-cache-dir /media/v100/disk3t/skating/fs1000_static_videomae_c1_top4_cross_attention \
  --static-cache-prefix static_videomae_c1_top4_cross_attention \
  --static-feature-dim 6914 \
  --checkpoint /media/v100/disk3t/skating/experiments/videomae_c1_top4_cross_attention/seed2026_warmstart/best_spearman.pth
```

To use another GPU, replace `cuda:0` in precomputation and `--gpu 0` in training/evaluation with the same desired device index.

## References

- Perrett et al., "Temporal-Relational CrossTransformers for Few-Shot Action Recognition," CVPR 2021.
- Thatipelli et al., "Spatio-Temporal Relation Modeling for Few-Shot Action Recognition," CVPR 2022.
- Wang et al., "Hybrid Relation Guided Set Matching for Few-Shot Action Recognition," CVPR 2022.
