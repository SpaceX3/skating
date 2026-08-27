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

Build the FineFS train-only four-class bank:

```bash
bash scripts/run_fs1000_class_conditioned_retrieval.sh bank
```

Precompute the FS1000 `concat(k, m)` cache:

```bash
bash scripts/run_fs1000_class_conditioned_retrieval.sh cache
```

Train without rebuilding either cache:

```bash
bash scripts/run_fs1000_class_conditioned_retrieval.sh train
```

Evaluate both saved best checkpoints:

```bash
bash scripts/run_fs1000_class_conditioned_retrieval.sh eval
```

To use a different GPU or output directory:

```bash
GPU_UUID=GPU-... \
RUN_DIR=/media/v100/disk3t/skating/experiments/videomae_c1_class_retrieval/my_run \
bash scripts/run_fs1000_class_conditioned_retrieval.sh train
```

Expected generated locations:

- Bank: `/media/v100/disk3t/skating/finefs_c1_class_bank_first_token`
- FS1000 cache: `/media/v100/disk3t/skating/fs1000_static_videomae_c1_class_retrieval`
- Training log/checkpoints: `/media/v100/disk3t/skating/experiments/videomae_c1_class_retrieval/manual_seed2026`

The full FineFS bank is about 0.51 GB in float16. With the current 79,776 FS1000 timesteps, the 1536D float16 cache is about 0.25 GB.
