# FS1000 VideoMAE Static Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build frozen C1-selected VideoMAE static caches for the FS1000 benchmark and train the existing MRU scorer with those caches.

**Architecture:** A standalone precomputation module maps each MRU timestep to four half-second-grid C1 candidates, ensembles the three FineFS checkpoints, selects the most confident candidate, and mean-pools its first cliplet to 768 dimensions. The existing dataset and scorer consume the new `[T,768]` cache with only the static input dimension changed.

**Tech Stack:** Python 3, NumPy, PyTorch, existing `skating-action` environment, existing cached VideoMAE features.

## Global Constraints

- Use the 651/160 FS1000 benchmark split.
- Do not change or install packages into `skating-action`.
- Do not extract VideoMAE features again.
- Freeze and reuse FineFS C1 seeds 2026, 2027, and 2028.
- Keep MRU, AST, TimeSformer, MSE target, and training schedule unchanged.
- Write new caches and checkpoints to distinct paths.

---

### Task 1: Confidence-Selected Static Cache

**Files:**
- Create: `videomae_static.py`
- Create: `precompute_videomae_static.py`
- Create: `tests/test_videomae_static.py`

**Interfaces:**
- `candidate_indices(times, timestep_index) -> list[tuple[float, np.ndarray, np.ndarray]]`
- `select_static_sequence(features, times, dynamic_length, models, device) -> tuple[np.ndarray, dict]`
- `write_video_cache(...) -> dict`

- [ ] Write tests proving `t=2*i`, offsets `[0,.5,1,1.5]`, three prefix plus six context cliplets, max-softmax ensemble selection, `[8,768] -> [768]` mean pooling, incomplete-tail selection, and previous-vector fallback.
- [ ] Run `python -m unittest tests.test_videomae_static -v` and verify failure because `videomae_static` is absent.
- [ ] Implement only the tested selection and cache behavior.
- [ ] Run the focused tests and verify they pass.
- [ ] Commit the cache implementation.

### Task 2: Dataset And Scorer Integration

**Files:**
- Modify: `dataset/dataset_fs800.py`
- Modify: `main.py`
- Modify: `model.py`
- Create: `tests/test_videomae_integration.py`

**Interfaces:**
- `FeatureDatasetWithStaticCache(..., cache_dir_name="static_videomae_c1_cache", cache_prefix="static_videomae_c1")`
- `scoring_head(..., static_in_dim=768)`

- [ ] Write tests that load `[T,768]`, reject a length mismatch, run the scorer with 768-dimensional static input, and execute `main.py --help` without the old `dev` NameError.
- [ ] Run the tests and verify the expected failures.
- [ ] Change the cache defaults, static input dimension, GPU default, seed setup, and isolated output paths without changing MRU computation or optimization schedules.
- [ ] Run the integration tests and Python compilation.
- [ ] Commit the integration.

### Task 3: Cache Generation And Audit

**Files:**
- Generate: `/media/v100/disk3t/skating/fs1000_static_videomae_c1/*.npy`
- Generate: `/media/v100/disk3t/skating/experiments/videomae_static_c1/cache_report.json`

- [ ] Run the precomputation script with the existing FS1000 feature root and all three C1 checkpoints.
- [ ] Require 811 caches, exact `[T_dyn,768]` shapes, finite values, and complete train/validation coverage.
- [ ] Record selected offset counts, incomplete candidate groups, and previous-vector fallbacks.

### Task 4: FS1000 TES Training And Evaluation

**Files:**
- Generate: `/media/v100/disk3t/skating/experiments/videomae_static_c1/run_seed2026/`

- [ ] Run the 200-epoch TES training command in the existing environment and retain readable logs.
- [ ] Evaluate the best-Spearman and best-loss checkpoints on the 160-video validation split.
- [ ] Report validation Spearman, MSE, best epochs, and comparison with the archived ResNet result, clearly identifying that it is not a paired multi-seed ablation.
