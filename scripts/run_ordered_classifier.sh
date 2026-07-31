#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/v100/ZYQ/skating}"
FINEFS_ROOT="${FINEFS_ROOT:-/home/v100/ZYQ/FineFS}"
DINO_PY="${DINO_PY:-/home/v100/anaconda3/envs/skating-dinov2/bin/python}"
ACTION_PY="${ACTION_PY:-/home/v100/anaconda3/envs/skating-action/bin/python}"
PREPROCESS_GPU="${PREPROCESS_GPU:-0}"
TRAIN_GPU="${TRAIN_GPU:-1}"
SEED="${SEED:-2026}"
STATIC_FPS="${STATIC_FPS:-2}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-200}"
ORDERED_PROJECTION_DIM="${ORDERED_PROJECTION_DIM:-16}"
ORDERED_TIME_BASIS="${ORDERED_TIME_BASIS:-2}"
ORDERED_HIDDEN_DIM="${ORDERED_HIDDEN_DIM:-256}"

CACHE_PREFIX="${CACHE_PREFIX:-static_dinov2_cls_patch_mean}"
FINEFS_CACHE="${FINEFS_CACHE:-$FINEFS_ROOT/static_dinov2_cls_patch_mean_rag_cache}"
CORPUS="${CORPUS:-$PROJECT_ROOT/rag_artifacts/action_rag_corpus_ordered_t${SEQUENCE_LENGTH}.pt}"
SPLIT="${SPLIT:-$PROJECT_ROOT/rag_artifacts/finefs_semantic_split_ordered_t${SEQUENCE_LENGTH}.json}"
RESULTS="${RESULTS:-$PROJECT_ROOT/rag_results/semantic_ordered_t${SEQUENCE_LENGTH}}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_ordered_classifier.sh test
  bash scripts/run_ordered_classifier.sh preprocess
  bash scripts/run_ordered_classifier.sh build-corpus
  bash scripts/run_ordered_classifier.sh split
  bash scripts/run_ordered_classifier.sh train
  bash scripts/run_ordered_classifier.sh all

Useful overrides:
  PREPROCESS_GPU=0 TRAIN_GPU=1 STATIC_FPS=2 SEQUENCE_LENGTH=8
  BATCH_SIZE=256 EPOCHS=200 ORDERED_PROJECTION_DIM=16
  ORDERED_TIME_BASIS=2 ORDERED_HIDDEN_DIM=256 SEED=2026
EOF
}

run_test() {
  "$ACTION_PY" -m py_compile \
    action.py action_rag.py semantic_rag.py semantic_main.py \
    scripts/build_action_rag_corpus.py
  "$ACTION_PY" -m unittest discover -v tests
}

preprocess() {
  "$DINO_PY" action.py \
    --dataset_mode finefs \
    --root_path "$FINEFS_ROOT" \
    --cache_dir_name "$(basename "$FINEFS_CACHE")" \
    --cache_prefix "$CACHE_PREFIX" \
    --gpu "$PREPROCESS_GPU" \
    --infer_batch_size 16 \
    --decode_batch_size 128 \
    --frames_per_second "$STATIC_FPS" \
    --sample_first_sec 2.0 \
    --finefs_stride 2.0 \
    --save_frame_sequences \
    --seed "$SEED"
}

build_corpus() {
  mkdir -p "$(dirname "$CORPUS")"
  "$ACTION_PY" scripts/build_action_rag_corpus.py \
    --finefs-root "$FINEFS_ROOT" \
    --fs1000-root "/home/v100/ZYQ/FS1000 Dataset" \
    --cache-dir "$FINEFS_CACHE" \
    --cache-prefix "$CACHE_PREFIX" \
    --frame-cache-prefix "${CACHE_PREFIX}_frames" \
    --include-ordered-sequences \
    --ordered-sequence-length "$SEQUENCE_LENGTH" \
    --max-background-per-video 3 \
    --output "$CORPUS"
}

split_corpus() {
  if [[ -f "$SPLIT" ]]; then
    echo "ordered semantic split already exists: $SPLIT"
    return
  fi
  "$ACTION_PY" semantic_main.py \
    --mode split \
    --corpus "$CORPUS" \
    --split-path "$SPLIT" \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --seed "$SEED"
}

train_classifier() {
  [[ -f "$SPLIT" ]] || split_corpus
  mkdir -p "$RESULTS"
  local run_name="ordered_classifier_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
  "$ACTION_PY" semantic_main.py \
    --mode train-classifier \
    --corpus "$CORPUS" \
    --split-path "$SPLIT" \
    --output-dir "$RESULTS" \
    --run-name "$run_name" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --lr 3e-4 \
    --weight-decay 5e-6 \
    --step-size 20 \
    --gamma 0.5 \
    --evidence-dim 256 \
    --ordered-projection-dim "$ORDERED_PROJECTION_DIM" \
    --ordered-time-basis "$ORDERED_TIME_BASIS" \
    --ordered-hidden-dim "$ORDERED_HIDDEN_DIM" \
    --dropout 0.1 \
    --gpu "$TRAIN_GPU" \
    --seed "$SEED"
}

case "${1:-}" in
  test) run_test ;;
  preprocess) preprocess ;;
  build-corpus) build_corpus ;;
  split) split_corpus ;;
  train) train_classifier ;;
  all)
    run_test
    preprocess
    build_corpus
    split_corpus
    train_classifier
    ;;
  *) usage; exit 1 ;;
esac
