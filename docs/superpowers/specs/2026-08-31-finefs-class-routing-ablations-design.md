# FineFS Class-Routing Ablations Design

## Context

The current FineFS Top-4 Cross-Attention experiment uses the frozen C1 coarse
classifier twice at every 2-second dynamic timestep:

1. It selects the highest-confidence 5-second candidate window from offsets
   `0`, `0.5`, `1.0`, and `1.5` seconds.
2. It routes the selected query to the two classes with the highest C1
   probabilities.

This change isolates the second use. Candidate-window selection remains
unchanged, while class routing is replaced with either the FineFS ground-truth
class or a uniformly random class.

## Experiments

### Ground-truth routing

For each dynamic timestep, retain the query selected by the existing C1
ensemble. Use the selected candidate's `first_index` to locate the corresponding
row in `cliplet_manifest.jsonl` by `(routine_id, cliplet_index)`. Convert its
label to one of the four existing coarse retrieval classes:

- `background` -> `background`
- `jump`, `jump_1`, `jump_2`, `jump_3`, `jump_4` -> `jump`
- `spin` -> `spin`
- `sequence` -> `sequence`

Route only to that class and retrieve its four nearest train-bank vectors.
This is an oracle ablation at validation and test time; it is not a deployable
inference configuration.

### Random routing

For each dynamic timestep, sample one of the four coarse classes uniformly.
Sampling uses a dedicated NumPy generator with seed `2026` by default, so the
precomputed cache is deterministic and independent of training data-loader
ordering. The random class may equal the ground-truth class by chance.

Route only to the sampled class and retrieve its four nearest train-bank
vectors.

## Invariants

Both ablations preserve all components after class routing:

- The C1 ensemble still chooses one candidate window from offsets
  `[0, 0.5, 1.0, 1.5]`.
- The query remains the first VideoMAE token of the selected candidate's first
  cliplet.
- Retrieval remains cosine Top-4 within the chosen FineFS train-only class
  bank.
- Cross-Attention keeps the same projection dimensions, four heads, residual
  query connection, static projection, and temporal score head.
- AST and Timesformer inputs, FineFS E11 split, TES target, optimizer schedule,
  and two-stage training remain unchanged.

Because each ablation routes one class instead of two, the static cache layout
is:

```text
query (768) + supports (1 x 4 x 768) + class weight (1) = 3841 dimensions
```

The model must therefore be constructed with `top_classes=1`, `top_k=4`, and
`static_feature_dim=3841`. Ground-truth and random experiments have identical
architecture and cache dimensions, so their comparison isolates routing
quality.

The existing C1 Top-2 run remains the reference result, but its `6914D` cache
and two-class aggregation make it a related baseline rather than a strictly
capacity-matched one-class comparison.

## Data Flow

1. Load all FineFS video IDs from E11 `video_split.json`.
2. Load aligned AST and Timesformer features and set the dynamic length to
   their minimum length.
3. Run the unchanged C1 candidate selector over existing VideoMAE features.
4. Record the selected first cliplet index at each dynamic timestep.
5. Produce one routed class per timestep using `ground_truth` or `random` mode.
6. Convert the routed class to a one-hot four-class probability vector.
7. Reuse the existing class-conditioned retriever with `top_classes=1` and
   `top_k=4`.
8. Pack and save an independent `3841D` float16 cache for each routing mode.
9. Train or test the existing scorer with `top_classes=1`.

If the existing selector falls back to the previous query because a later
timestep has no complete candidate group, ground-truth routing also repeats the
previous selected cliplet index and class. The first timestep already fails in
the existing selector when no complete candidate exists, so no new fallback is
introduced. Random routing remains one independently sampled, deterministic
class per dynamic timestep.

## Interfaces and Outputs

The FineFS precomputation command gains:

- `--routing ground_truth|random`
- `--manifest` for ground-truth label lookup
- `--random-seed`, default `2026`

Each routing mode uses a separate cache directory and prefix:

- Ground truth:
  `finefs_static_videomae_gt_top4_cross_attention` and
  `static_videomae_gt_top4_cross_attention`
- Random:
  `finefs_static_videomae_random_top4_cross_attention` and
  `static_videomae_random_top4_cross_attention`

The training entry point gains `--top-classes`, defaulting to `2` so the
existing FineFS C1 Top-2 command remains valid. The two ablations explicitly
pass `--top-classes 1 --static-feature-dim 3841`.

The cache report records routing mode, seed where applicable, selected class
counts, layout, and feature dimension.

## Verification Scope

No full cache generation or training is run. Verification is limited to:

- Python 3.8 syntax compilation.
- A small deterministic routing check for GT label mapping and seeded random
  routing.
- Model construction with `top_classes=1` and a `3841D` cache contract.
- CLI help and command documentation consistency.

The repository manual-run document is extended with explicit precompute,
training, and test commands for both ablations.
