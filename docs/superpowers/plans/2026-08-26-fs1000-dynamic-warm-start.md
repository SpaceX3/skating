# FS1000 Dynamic Warm-Start Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Warm-start the FS1000 first-token experiment from the best ResNet run's dynamic MRU weights while randomly initializing the new static projection and fusion head.

**Architecture:** Add a focused state-dict loader in `main.py` that excludes `static_proj.*` and `time_score_mlp.*`, validates all remaining names and dimensions, then loads them before training. Keep the existing two-stage schedule and expose the source through the CLI and manual runner.

**Tech Stack:** Python 3.8, PyTorch, unittest, Bash

## Global Constraints

- Do not modify `/home/v100/anaconda3/envs/skating-action`.
- Do not start a full training run automatically.
- Use seed 2026 and the existing first-token cache.
- Load exactly the dynamic branch; initialize `static_proj` and `time_score_mlp` randomly.

---

### Task 1: Dynamic Checkpoint Loader

**Files:**
- Modify: `main.py`
- Test: `tests/test_dynamic_warm_start.py`

**Interfaces:**
- Consumes: a current `scoring_head` and a raw PyTorch state dict path
- Produces: `load_dynamic_checkpoint(model, checkpoint_path) -> list[str]`

- [ ] **Step 1: Write the failing test**

Create a source state with changed dynamic values and incompatible legacy
`static_proj` values. Verify that the loader changes all dynamic tensors while
leaving `static_proj.*` and `time_score_mlp.*` unchanged.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/v100/anaconda3/envs/skating-action/bin/python -m unittest tests.test_dynamic_warm_start -v
```

Expected: import failure because `load_dynamic_checkpoint` does not exist.

- [ ] **Step 3: Write minimal implementation**

Add `load_dynamic_checkpoint`, exclude the two specified prefixes, validate the
34 expected dynamic keys and shapes, call `model.load_state_dict(...,
strict=False)`, and return the loaded key names.

- [ ] **Step 4: Run test to verify it passes**

Run the command from Step 2 and expect one passing test.

### Task 2: Training Entry Point

**Files:**
- Modify: `main.py`
- Modify: `scripts/run_fs1000_videomae_static.sh`

**Interfaces:**
- Consumes: `--init-dynamic-checkpoint PATH`
- Produces: logged initialization summary before stage-1 training

- [ ] **Step 1: Add the CLI argument**

Default the Python argument to `None` and invoke the loader after seeded model
construction when a path is supplied.

- [ ] **Step 2: Update the manual runner**

Default `INIT_DYNAMIC_CHECKPOINT` to
`/home/v100/ZYQ/skating/fs800_result/checkpoint_best_0.872.pth`, pass it to
`main.py`, and default `RUN_DIR` to a new `dynamic_init` experiment directory.

- [ ] **Step 3: Run focused regression tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/v100/anaconda3/envs/skating-action/bin/python -m unittest tests.test_dynamic_warm_start tests.test_videomae_integration tests.test_videomae_static -v
bash -n scripts/run_fs1000_videomae_static.sh
```

Expected: all tests pass and the shell script parses successfully.

- [ ] **Step 4: Commit**

Commit the specification, plan, loader, test, and manual runner on
`experiment/fs1000-videomae-c1-first-token`.
