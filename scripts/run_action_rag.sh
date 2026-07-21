#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/v100/ZYQ/skating}"
FS1000_ROOT="${FS1000_ROOT:-/home/v100/ZYQ/FS1000 Dataset}"
FINEFS_ROOT="${FINEFS_ROOT:-/home/v100/ZYQ/FineFS}"
DINO_PY="${DINO_PY:-/home/v100/anaconda3/envs/skating-dinov2/bin/python}"
ACTION_PY="${ACTION_PY:-/home/v100/anaconda3/envs/skating-action/bin/python}"
PREPROCESS_GPU="${PREPROCESS_GPU:-0}"
TRAIN_GPU="${TRAIN_GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-2026}"
TOP_K="${TOP_K:-8}"
FINEFS_CACHE="${FINEFS_CACHE:-$FINEFS_ROOT/static_dinov2_cls_patch_mean_rag_cache}"
CORPUS="${CORPUS:-$PROJECT_ROOT/rag_artifacts/action_rag_corpus.pt}"
CANDIDATES="${CANDIDATES:-$PROJECT_ROOT/rag_artifacts/candidates}"
RESULTS="${RESULTS:-$PROJECT_ROOT/rag_results}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$PROJECT_ROOT/fs800_result/checkpoint_epoch42_loss112.53_spear0.879.pth}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_action_rag.sh test
  bash scripts/run_action_rag.sh fs1000-times
  bash scripts/run_action_rag.sh finefs-features
  bash scripts/run_action_rag.sh build-corpus
  bash scripts/run_action_rag.sh audit-retrieval
  bash scripts/run_action_rag.sh candidates
  bash scripts/run_action_rag.sh preprocess-all
  bash scripts/run_action_rag.sh train-dynamic
  DYNAMIC_CHECKPOINT=/path/to/dynamic_best.pth bash scripts/run_action_rag.sh train-rag
  CHECKPOINT=/path/to/checkpoint.pth bash scripts/run_action_rag.sh evaluate
  CHECKPOINT=/path/to/rag_checkpoint.pth bash scripts/run_action_rag.sh evaluate-baseline

Optional environment variables:
  PREPROCESS_GPU=0 TRAIN_GPU=1 NUM_WORKERS=8 SEED=2026 TOP_K=8
  INIT_CHECKPOINT=/path/to/legacy_checkpoint.pth
  DYNAMIC_CHECKPOINT=/path/to/dynamic_best.pth
  CHECKPOINT=/path/to/evaluation_checkpoint.pth
EOF
}

run_test() {
  "$ACTION_PY" -m py_compile \
    model.py action_rag.py main.py eval.py action.py \
    dataset/dataset_fs800.py \
    scripts/build_action_rag_corpus.py \
    scripts/precompute_action_candidates.py \
    scripts/audit_action_retrieval.py
  "$ACTION_PY" -m unittest -v tests/test_action_rag.py
}

fs1000_times() {
  "$DINO_PY" action.py \
    --dataset_mode fs1000 \
    --root_path "$FS1000_ROOT" \
    --cache_dir_name static_dinov2_cls_patch_mean_cache \
    --cache_prefix static_dinov2_cls_patch_mean \
    --times_only
}

finefs_features() {
  "$DINO_PY" action.py \
    --dataset_mode finefs \
    --root_path "$FINEFS_ROOT" \
    --cache_dir_name "$(basename "$FINEFS_CACHE")" \
    --cache_prefix static_dinov2_cls_patch_mean \
    --gpu "$PREPROCESS_GPU" \
    --infer_batch_size 16 \
    --decode_batch_size 128 \
    --finefs_stride 2.0
}

build_corpus() {
  mkdir -p "$(dirname "$CORPUS")"
  "$ACTION_PY" scripts/build_action_rag_corpus.py \
    --finefs-root "$FINEFS_ROOT" \
    --fs1000-root "$FS1000_ROOT" \
    --cache-dir "$FINEFS_CACHE" \
    --cache-prefix static_dinov2_cls_patch_mean \
    --max-background-per-video 3 \
    --output "$CORPUS"
}

audit_retrieval() {
  "$ACTION_PY" scripts/audit_action_retrieval.py \
    --corpus "$CORPUS" \
    --top-k "$TOP_K" \
    --device "cuda:$PREPROCESS_GPU"
}

precompute_candidates() {
  mkdir -p "$CANDIDATES"
  "$ACTION_PY" scripts/precompute_action_candidates.py \
    --fs1000-root "$FS1000_ROOT" \
    --finefs-root "$FINEFS_ROOT" \
    --query-cache-dir "$FS1000_ROOT/static_dinov2_cls_patch_mean_cache" \
    --query-cache-prefix static_dinov2_cls_patch_mean \
    --corpus "$CORPUS" \
    --output-dir "$CANDIDATES" \
    --top-k "$TOP_K" \
    --dedup-pool-size 64 \
    --split all \
    --device "cuda:$PREPROCESS_GPU"
}

train_dynamic() {
  local run_name="dynamic_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$RESULTS/dynamic"
  "$ACTION_PY" main.py \
    --training-stage dynamic \
    --root-path "$FS1000_ROOT" \
    --static-cache-dir-name static_dinov2_cls_patch_mean_cache \
    --static-cache-prefix static_dinov2_cls_patch_mean \
    --init-checkpoint "$INIT_CHECKPOINT" \
    --output-dir "$RESULTS/dynamic" \
    --run-name "$run_name" \
    --batch-size 8 \
    --num-workers "$NUM_WORKERS" \
    --epochs 120 \
    --lr 1e-4 \
    --gpu "$TRAIN_GPU" \
    --seed "$SEED"
  echo "The metric-stamped dynamic best checkpoint is printed above."
}

train_rag() {
  : "${DYNAMIC_CHECKPOINT:?Set DYNAMIC_CHECKPOINT to the dynamic-stage best checkpoint}"
  local run_name="rag_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$RESULTS/rag"
  "$ACTION_PY" main.py \
    --training-stage rag \
    --root-path "$FS1000_ROOT" \
    --static-cache-dir-name static_dinov2_cls_patch_mean_cache \
    --static-cache-prefix static_dinov2_cls_patch_mean \
    --rag-corpus-path "$CORPUS" \
    --candidate-dir "$CANDIDATES" \
    --dynamic-checkpoint "$DYNAMIC_CHECKPOINT" \
    --output-dir "$RESULTS/rag" \
    --run-name "$run_name" \
    --batch-size 4 \
    --num-workers "$NUM_WORKERS" \
    --epochs 100 \
    --lr 3e-4 \
    --delta-l2-weight 1e-4 \
    --gpu "$TRAIN_GPU" \
    --seed "$SEED"
  echo "The metric-stamped RAG best checkpoint is printed above."
}

evaluate() {
  : "${CHECKPOINT:?Set CHECKPOINT to a dynamic or RAG checkpoint}"
  "$ACTION_PY" eval.py \
    --checkpoint "$CHECKPOINT" \
    --root-path "$FS1000_ROOT" \
    --rag-corpus-path "$CORPUS" \
    --candidate-dir "$CANDIDATES" \
    --batch-size 4 \
    --num-workers "$NUM_WORKERS" \
    --gpu "$TRAIN_GPU"
}

evaluate_baseline() {
  : "${CHECKPOINT:?Set CHECKPOINT to a RAG checkpoint}"
  "$ACTION_PY" eval.py \
    --checkpoint "$CHECKPOINT" \
    --root-path "$FS1000_ROOT" \
    --rag-corpus-path "$CORPUS" \
    --candidate-dir "$CANDIDATES" \
    --batch-size 4 \
    --num-workers "$NUM_WORKERS" \
    --gpu "$TRAIN_GPU" \
    --force-baseline-only
}

case "${1:-}" in
  test) run_test ;;
  fs1000-times) fs1000_times ;;
  finefs-features) finefs_features ;;
  build-corpus) build_corpus ;;
  audit-retrieval) audit_retrieval ;;
  candidates) precompute_candidates ;;
  preprocess-all)
    fs1000_times
    finefs_features
    build_corpus
    audit_retrieval
    precompute_candidates
    ;;
  train-dynamic) train_dynamic ;;
  train-rag) train_rag ;;
  evaluate) evaluate ;;
  evaluate-baseline) evaluate_baseline ;;
  evaluate-dynamic) evaluate_baseline ;;
  *) usage; exit 2 ;;
esac
