# Random-Class Top-4 Score Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible random-one-class routing ablation while preserving C1 query-window selection and all downstream Top-4 score-conditioned retrieval behavior.

**Architecture:** A small pure NumPy routing helper creates one-hot class probabilities per video from seed `2026` and video ordinal. The existing precomputation script selects the query cliplet with C1 exactly as before, then substitutes these probabilities only in `random-one` mode. Training and evaluation receive the matching `top_classes=1` model configuration and consume a separate 3857-dimensional cache.

**Tech Stack:** Python 3.8, NumPy, PyTorch, existing `unittest` suite.

## Global Constraints

- Do not start a full cache build or training run.
- Keep `c1-top2` as the default routing mode and preserve its 6946-dimensional cache compatibility.
- Sample uniformly from `background`, `jump`, `spin`, and `sequence` once during precomputation.
- Keep C1 offset selection over `[t, t+0.5, t+1.0, t+1.5]` unchanged.
- Keep Top-4 cosine retrieval, score descriptors, masks, cross-attention, MRU, optimizer, and evaluation unchanged.
- Write explicit manual commands to a repository Markdown document.

---

### Task 1: Deterministic Random-One Routing

**Files:**
- Create: `routing_ablation.py`
- Create: `tests/test_random_class_routing.py`

**Interfaces:**
- Produces: `random_one_class_probabilities(num_queries: int, num_classes: int, seed: int, video_ordinal: int) -> np.ndarray` with shape `[num_queries, num_classes]` and `float32` one-hot rows.
- Consumes: no project state; only scalar routing parameters.

- [ ] **Step 1: Write the failing routing tests**

Test that repeated calls with `(64, 4, 2026, 3)` are identical, every row sums to one, every entry is zero or one, and all selected indices fall in `[0, 3]`. Test invalid non-positive query/class counts raise `ValueError`.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
/home/v100/anaconda3/envs/skating-action/bin/python -m unittest \
  tests.test_random_class_routing
```

Expected: import failure because `routing_ablation.py` does not exist.

- [ ] **Step 3: Implement the routing helper**

Use a per-video generator so partial cache rebuilds do not change other videos:

```python
sequence = np.random.SeedSequence([int(seed), int(video_ordinal)])
generator = np.random.default_rng(sequence)
indices = generator.integers(0, int(num_classes), size=int(num_queries))
probabilities = np.zeros((num_queries, num_classes), dtype=np.float32)
probabilities[np.arange(num_queries), indices] = 1.0
```

- [ ] **Step 4: Re-run the focused test and verify GREEN**

Run the command from Step 2. Expected: all routing tests pass.

### Task 2: Parameterize Top-4 Cache Precomputation

**Files:**
- Modify: `precompute_top4_cross_attention_static.py`
- Modify: `tests/test_top4_cross_attention.py`

**Interfaces:**
- Consumes: `random_one_class_probabilities(...)` from Task 1.
- Produces: CLI options `--routing-mode {c1-top2,random-one}` and `--routing-seed INT`.
- Produces: `cache_dimension(top_classes: int) -> int`, yielding `6946` for 2 and `3857` for 1.

- [ ] **Step 1: Write failing cache-configuration tests**

Assert that `cache_dimension(2) == 6946` and `cache_dimension(1) == 3857`. Pack a one-class cache with query `[B,768]`, supports `[B,1,4,768]`, scores `[B,1,4,3]`, masks `[B,1,4]`, and weights `[B,1]`, then assert its final dimension is `3857`.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
/home/v100/anaconda3/envs/skating-action/bin/python -m unittest \
  tests.test_random_class_routing \
  tests.test_top4_cross_attention
```

Expected: failure because `cache_dimension` and routing CLI behavior are absent.

- [ ] **Step 3: Implement routing-mode configuration**

Keep `select_static_sequence_and_probabilities(...)` unchanged. Immediately after it returns, replace only `probabilities` in random mode:

```python
if args.routing_mode == "random-one":
    probabilities = random_one_class_probabilities(
        len(query), len(CLASS_NAMES), args.routing_seed, ordinal
    )
    top_classes = 1
else:
    top_classes = 2
```

Pass `top_classes` into `retriever.retrieve`. Derive the cache dimension and static layout from `top_classes`. Use prefix `static_videomae_random1_top4_score_attention` and the default directory `/media/v100/disk3t/skating/fs1000_static_videomae_random1_top4_score_attention` in random mode. Record `routing_mode`, `routing_seed`, and selected class counts in `cache_report.json`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all tests pass without creating a full cache.

### Task 3: Configure Training, Evaluation, And Manual Commands

**Files:**
- Modify: `main.py`
- Modify: `eval.py`
- Modify: `tests/test_top4_cross_attention.py`
- Create: `docs/FS1000_RANDOM_CLASS_TOP4_SCORE_ATTENTION.md`

**Interfaces:**
- Consumes: random cache dimension `3857` and one-class layout from Task 2.
- Produces: `build_model(static_in_dim: int = 6946, top_classes: int = 2)`.
- Produces: `--top-classes INT` in both training and evaluation.

- [ ] **Step 1: Write the failing one-class model test**

Build `main.build_model(static_in_dim=3857, top_classes=1)` and assert the model's `query_support_cross_attention.top_classes == 1`, `top_k == 4`, `score_dim == 3`, and `static_proj.in_features == 768`. Run a finite forward/backward pass through `QuerySupportCrossAttention` with a one-class packed cache.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
/home/v100/anaconda3/envs/skating-action/bin/python -m unittest \
  tests.test_top4_cross_attention
```

Expected: `build_model` rejects the new `top_classes` argument.

- [ ] **Step 3: Thread `top_classes` through training and evaluation**

Change `build_model` to pass `top_classes` into `scoring_head`. Add `--top-classes` with default `2` to `main.py` and `eval.py`, and use it when constructing the model. Do not alter optimizer, stage freezing, checkpoint loading, loss, or score index behavior.

- [ ] **Step 4: Write the manual experiment document**

Document explicit commands for:

- random-one cache precomputation with `--routing-mode random-one --routing-seed 2026`;
- warm-start training with `--static-feature-dim 3857 --top-classes 1`;
- fully random training without `--init-dynamic-checkpoint` and with `--freeze-backbone-epochs 0`;
- evaluation with `--static-feature-dim 3857 --top-classes 1`.

Use new cache and experiment directories. State that no full run was launched.

- [ ] **Step 5: Run the complete focused suite**

```bash
/home/v100/anaconda3/envs/skating-action/bin/python -m unittest \
  tests.test_finefs_class_bank \
  tests.test_class_conditioned_retrieval \
  tests.test_random_class_routing \
  tests.test_top4_cross_attention
```

Expected: all focused tests pass.

- [ ] **Step 6: Commit and push the implementation**

```bash
git add routing_ablation.py precompute_top4_cross_attention_static.py \
  main.py eval.py tests/test_random_class_routing.py \
  tests/test_top4_cross_attention.py \
  docs/FS1000_RANDOM_CLASS_TOP4_SCORE_ATTENTION.md
git commit -m "feat: add random-class Top-4 score ablation"
HTTPS_PROXY=http://127.0.0.1:12000 \
HTTP_PROXY=http://127.0.0.1:12000 \
git push
```

