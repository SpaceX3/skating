# FS1000 Random-Class Top-4 Score Attention Design

## Objective

Measure whether the FineFS C1 coarse-class routing contributes useful information to
the score-conditioned Top-4 retrieval experiment. Replace only the current C1 Top-2
class route with one uniformly random coarse class. Keep query selection, retrieval,
score conditioning, MRU scoring, data split, optimization, and evaluation unchanged.

FS1000 has no compatible per-timestep coarse-class ground truth, so the originally
considered ground-truth routing ablation is excluded.

## Compared Conditions

- Existing baseline: C1 routes each query to its Top-2 classes.
- New ablation: each query is routed to one class sampled uniformly from
  `background`, `jump`, `spin`, and `sequence`.

The random route is fixed during cache precomputation with seed `2026`. It is not
resampled between training epochs. The class for a timestep is derived reproducibly
from the seed, video ordinal, and timestep position, so rebuilding only part of the
cache does not change existing assignments.

## Preserved Behavior

C1 continues to evaluate the four offsets `[t, t+0.5, t+1.0, t+1.5]` and select the
highest-confidence query cliplet. Only the class probabilities used for FineFS bank
routing are replaced after query selection.

Within the randomly selected class, cosine retrieval still selects Top-4 FineFS
train-bank supports. Each support retains its VideoMAE feature, normalized
`[BV, GOE, panel_score]` descriptor, and score-valid mask. The score-aware
cross-attention continues to use visual support features as Keys and visual plus
score information as Values. The MRU dynamic branch and temporal scoring head are
unchanged.

## Interface And Cache

The existing precomputation entry point gains a routing option while retaining the
current Top-2 behavior as its default:

```text
--routing-mode c1-top2 | random-one
--routing-seed 2026
```

Training and evaluation gain `--top-classes`, defaulting to `2` for backward
compatibility with the existing experiment. The random ablation passes
`--top-classes 1`.

The one-class cache layout is:

```text
query 768
+ supports 1 x 4 x 768
+ normalized scores 1 x 4 x 3
+ score masks 1 x 4
+ class weight 1
= 3857 values per timestep
```

The random cache uses a new prefix and output directory, so it cannot overwrite or
be confused with the existing 6946-dimensional Top-2 cache.

## Implementation Choice

Use one parameterized precomputation path rather than a copied script. This keeps
the Top-4 retrieval and cache packing identical across the baseline and ablation.
A separate script would duplicate the critical retrieval path. Resampling random
classes during every epoch is rejected because it changes the training distribution
between epochs and would require online retrieval instead of the established cache.

## Verification And Execution

Focused tests will verify deterministic random routing, valid class indices, the
3857-dimensional cache layout, and a finite one-class model forward/backward pass.
No full cache build or training run will be started automatically.

The implementation will include a user-facing Markdown file with explicit commands
for:

1. Precomputing the random-class cache.
2. Training from the existing MRU checkpoint.
3. Training all parameters from random initialization.
4. Evaluating a saved checkpoint.

