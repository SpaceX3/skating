# FS1000 VideoMAE Static Branch Design

## Goal

Replace the `best` branch's per-timestep ResNet50 static features with frozen VideoMAE features selected by the FineFS C1 confidence selector, while leaving MRU dynamic inputs, fusion, score target, loss, and training schedule unchanged.

## Dataset And Alignment

- Use the existing FS1000 benchmark split: 651 training and 160 validation videos.
- Preserve AST and TimeSformer inputs and define `T_dyn = min(T_ast, T_video)` exactly as the existing loader does.
- For dynamic timestep `i`, set `t = 2 * i` seconds.
- Form up to four complete five-second candidates starting at `t`, `t+0.5`, `t+1.0`, and `t+1.5`.
- Run the frozen FineFS C1 seeds 2026, 2027, and 2028 on every complete candidate. Average their logits and select the candidate with the largest maximum softmax probability.
- Take the selected candidate's first one-second VideoMAE cliplet. Average its eight temporal tokens to one 768-dimensional vector.
- Near the video tail, select among the complete candidates that exist. If none exist, repeat the previous timestep's selected vector.
- Save one float32 cache per video with shape `[T_dyn, 768]`.

The FineFS selector's 85.21% accuracy describes FineFS only. FS1000 has no compatible coarse-action labels, so this experiment will not claim a selector accuracy on FS1000.

## Scoring Model

Keep the existing bidirectional MRU path unchanged. Its per-timestep 512-dimensional feature is concatenated with `static_proj(VideoMAE)`, where only `static_in_dim` changes from 2048 to 768 and the projected size remains 128. The time-scoring MLP therefore remains `640 -> 256 -> 128 -> 1`, followed by the existing temporal mean.

Train TES only on the original split with the existing settings: batch size 16, MSE, 200 epochs, first 30 epochs training only `static_proj` and `time_score_mlp`, then full-model training, Adam learning rate `1e-4`, weight decay `5e-6`, and the existing StepLR schedules.

## Operational Changes

- Add a precomputation script that consumes existing FS1000 VideoMAE caches and the three frozen FineFS C1 checkpoints. It does not decode videos or retrain C1.
- Add a distinct dataset cache name and experiment output directory.
- Fix the existing `main.py --gpu` default-value error so the experiment can start.
- Save stable best-validation-loss and best-validation-Spearman checkpoints without overwriting the prior ResNet experiment.
- Record cache selection counts, tail fallbacks, resolved paths, seed, and epoch logs.

## Verification

Tests cover candidate construction, ensemble confidence selection, eight-token mean pooling, incomplete-tail selection, no-candidate repetition, cache shape, loader integration, 768-dimensional static projection, and CLI startup. A cache audit must cover all 811 split videos before training begins.
