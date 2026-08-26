# FS1000 VideoMAE First-Token Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a controlled FS1000 AQA ablation that stores token 0 of the C1-selected first VideoMAE cliplet instead of averaging its eight temporal tokens.

**Architecture:** Keep C1 candidate construction and ensemble selection unchanged. Change only the selected cliplet reduction, write the resulting `[T,768]` arrays under a new cache namespace, and point the unchanged MRU training pipeline at that cache.

**Tech Stack:** Python 3.8, NumPy, PyTorch, unittest, existing `skating-action` and `skating-action-e10` environments.

## Global Constraints

- Use the existing 651/160 FS1000 split and training seed 2026.
- Do not decode videos, extract VideoMAE again, install packages, or modify either conda environment.
- Keep the frozen FineFS C1 seeds 2026/2027/2028, candidate offsets, AST, TimeSformer, MRU, loss, and 200-epoch schedule unchanged.
- Preserve the completed mean-eight-token cache and experiment outputs.

---

### Task 1: First-Token Cache Behavior And Namespace

**Files:**
- Modify: `tests/test_videomae_static.py`
- Modify: `videomae_static.py`
- Modify: `precompute_videomae_static.py`
- Modify: `dataset/dataset_fs800.py`
- Modify: `main.py`
- Modify: `eval.py`
- Modify: `scripts/run_fs1000_videomae_static.sh`

**Interfaces:**
- Consumes: `select_static_sequence(features, times, dynamic_length, models, device, batch_size)`.
- Produces: a float32 `[T_dynamic,768]` array whose selected rows equal `features[first_index, 0]`.
- Produces: cache files named `static_videomae_c1_first_token_<video_id>_T<T>.npy`.

- [ ] **Step 1: Change the focused test to require token 0**

```python
def test_selects_candidate_and_takes_its_first_temporal_token(self):
    # Keep the existing candidate setup and selected index assertion.
    np.testing.assert_allclose(
        sequence[0], features[expected_first_index, 0]
    )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python \
  -m unittest tests.test_videomae_static.StaticSequenceTests.test_selects_candidate_and_takes_its_first_temporal_token -v
```

Expected: FAIL because the current implementation returns the mean of all eight temporal tokens.

- [ ] **Step 3: Implement only index-0 pooling**

Replace the selected-vector assignment in `videomae_static.py` with:

```python
sequence[timestep_index] = np.asarray(
    features[candidate["first_index"], 0], dtype=np.float32
)
```

Change cache metadata to `first_temporal_token`, cache directory defaults to `/media/v100/disk3t/skating/fs1000_static_videomae_c1_first_token`, prefix defaults to `static_videomae_c1_first_token`, run directory defaults to `/media/v100/disk3t/skating/experiments/videomae_static_c1_first_token/run_seed2026`, and the manual script seed to 2026.

- [ ] **Step 4: Verify GREEN and the complete suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/v100/anaconda3/envs/skating-action/bin/python \
  -W error::ResourceWarning -m unittest \
  tests.test_videomae_integration tests.test_videomae_static -v
bash -n scripts/run_fs1000_videomae_static.sh
```

Expected: 9 tests pass and the shell script has valid syntax.

- [ ] **Step 5: Commit the behavior change**

```bash
git add tests/test_videomae_static.py videomae_static.py \
  precompute_videomae_static.py dataset/dataset_fs800.py main.py eval.py \
  scripts/run_fs1000_videomae_static.sh
git commit -m "feat: use first VideoMAE temporal token"
```

### Task 2: Cache, Train, And Evaluate

**Files:**
- Generate: `/media/v100/disk3t/skating/fs1000_static_videomae_c1_first_token/`
- Generate: `/media/v100/disk3t/skating/experiments/videomae_static_c1_first_token/run_seed2026/`

**Interfaces:**
- Consumes: existing FS1000 VideoMAE features and frozen FineFS C1 checkpoints.
- Produces: audited first-token cache, 200-epoch log, `best_spearman.pth`, `best_loss.pth`, and `last.pth`.

- [ ] **Step 1: Generate the independent cache**

```bash
CUDA_VISIBLE_DEVICES=GPU-f40ff723-535e-10df-74d9-4b38ebeac3c5 \
  /home/v100/anaconda3/envs/skating-action-e10/bin/python \
  precompute_videomae_static.py --device cuda:0 --batch-size 512
```

- [ ] **Step 2: Audit all cache arrays**

For every ID in `train_fs800.txt` and `val_fs800.txt`, require the cache file to exist, have shape `[min(T_AST,T_TimeSformer),768]`, and contain only finite values. Require 811 checked videos and zero errors before training.

- [ ] **Step 3: Run the paired 200-epoch experiment**

```bash
CUDA_VISIBLE_DEVICES=GPU-f40ff723-535e-10df-74d9-4b38ebeac3c5 \
  /home/v100/anaconda3/envs/skating-action/bin/python main.py \
  --gpu 0 --seed 2026 --epochs 200 --batch-size 16 --num-workers 8 \
  --log-dir /media/v100/disk3t/skating/experiments/videomae_static_c1_first_token/run_seed2026
```

- [ ] **Step 4: Independently evaluate both best checkpoints**

Run `eval.py` once with `best_spearman.pth` and once with `best_loss.pth`. Record validation Spearman and MSE from the 160-video validation split.

- [ ] **Step 5: Compare against mean-eight-token pooling**

Report best epochs and deltas against Spearman `0.8472053656531183` and MSE `74.6912654876709`. Also report the final epoch to expose overfitting rather than presenting only the best checkpoint.
