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
SEMANTIC_NUM_WORKERS="${SEMANTIC_NUM_WORKERS:-0}"
SEMANTIC_BATCH_SIZE="${SEMANTIC_BATCH_SIZE:-512}"
SEED="${SEED:-2026}"
TOP_K="${TOP_K:-8}"
FINEFS_CACHE="${FINEFS_CACHE:-$FINEFS_ROOT/static_dinov2_cls_patch_mean_rag_cache}"
CORPUS="${CORPUS:-$PROJECT_ROOT/rag_artifacts/action_rag_corpus.pt}"
CANDIDATES="${CANDIDATES:-$PROJECT_ROOT/rag_artifacts/candidates_v2}"
RESULTS="${RESULTS:-$PROJECT_ROOT/rag_results}"
SEMANTIC_SPLIT="${SEMANTIC_SPLIT:-$PROJECT_ROOT/rag_artifacts/finefs_semantic_split.json}"
SEMANTIC_RESULTS="${SEMANTIC_RESULTS:-$RESULTS/semantic}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$PROJECT_ROOT/fs800_result/checkpoint_epoch42_loss112.53_spear0.879.pth}"
DYNAMIC_CHECKPOINT="${DYNAMIC_CHECKPOINT:-$PROJECT_ROOT/rag_results/dynamic/dynamic_seed2026_20260724_004400_best_epoch080_loss83.0444_spear0.8658.pth}"

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
  SEMANTIC_CHECKPOINT=/path/to/goe_v2.pth bash scripts/run_action_rag.sh candidates-v2
  bash scripts/run_action_rag.sh preprocess-all
  bash scripts/run_action_rag.sh semantic-split
  bash scripts/run_action_rag.sh train-semantic-classifier
  CLASSIFIER_CHECKPOINT=/path/to/classifier_v2.pth bash scripts/run_action_rag.sh train-semantic-goe
  SEMANTIC_CHECKPOINT=/path/to/semantic_best.pth bash scripts/run_action_rag.sh semantic-val
  SEMANTIC_CHECKPOINT=/path/to/semantic_best.pth bash scripts/run_action_rag.sh semantic-test
  bash scripts/run_action_rag.sh train-dynamic
  bash scripts/run_action_rag.sh train-rag
  CHECKPOINT=/path/to/checkpoint.pth bash scripts/run_action_rag.sh evaluate
  CHECKPOINT=/path/to/rag_checkpoint.pth bash scripts/run_action_rag.sh evaluate-baseline

Optional environment variables:
  PREPROCESS_GPU=0 TRAIN_GPU=1 NUM_WORKERS=8 SEED=2026 TOP_K=8
  SEMANTIC_BATCH_SIZE=512 SEMANTIC_NUM_WORKERS=0
  INIT_CHECKPOINT=/path/to/legacy_checkpoint.pth
  DYNAMIC_CHECKPOINT=/path/to/dynamic_best.pth  # optional override for train-rag
  CHECKPOINT=/path/to/evaluation_checkpoint.pth
  CLASSIFIER_CHECKPOINT=/path/to/classifier_v2.pth
  SEMANTIC_CHECKPOINT=/path/to/goe_v2.pth
EOF
}

run_test() {
  "$ACTION_PY" -m py_compile \
    model.py action_rag.py semantic_rag.py semantic_data.py semantic_main.py \
    main.py eval.py action.py \
    dataset/dataset_fs800.py \
    scripts/build_action_rag_corpus.py \
    scripts/precompute_action_candidates.py \
    scripts/audit_action_retrieval.py
  "$ACTION_PY" -m unittest discover -v tests
}

semantic_split() {
  if [[ -f "$SEMANTIC_SPLIT" ]]; then
    echo "FineFS semantic split already exists: $SEMANTIC_SPLIT"
    echo "Delete it explicitly or call semantic_main.py --overwrite-split to replace it."
    return
  fi
  "$ACTION_PY" semantic_main.py \
    --mode split \
    --corpus "$CORPUS" \
    --split-path "$SEMANTIC_SPLIT" \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --seed "$SEED"
}

train_semantic_classifier() {
  [[ -f "$SEMANTIC_SPLIT" ]] || semantic_split
  local run_name="semantic_classifier_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$SEMANTIC_RESULTS"
  "$ACTION_PY" semantic_main.py \
    --mode train-classifier \
    --corpus "$CORPUS" \
    --split-path "$SEMANTIC_SPLIT" \
    --output-dir "$SEMANTIC_RESULTS" \
    --run-name "$run_name" \
    --batch-size "$SEMANTIC_BATCH_SIZE" \
    --num-workers "$SEMANTIC_NUM_WORKERS" \
    --epochs 200 \
    --lr 3e-4 \
    --evidence-dim 256 \
    --encoder-hidden-dim 512 \
    --metadata-dim 64 \
    --dropout 0.1 \
    --gpu "$TRAIN_GPU" \
    --seed "$SEED"
}

train_semantic_goe() {
  [[ -f "$SEMANTIC_SPLIT" ]] || semantic_split
  : "${CLASSIFIER_CHECKPOINT:?Set CLASSIFIER_CHECKPOINT to the v2 classifier checkpoint}"
  local run_name="semantic_goe_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$SEMANTIC_RESULTS"
  "$ACTION_PY" semantic_main.py --mode train-goe --corpus "$CORPUS" \
    --split-path "$SEMANTIC_SPLIT" --classifier-checkpoint "$CLASSIFIER_CHECKPOINT" \
    --output-dir "$SEMANTIC_RESULTS" --run-name "$run_name" \
    --batch-size "$SEMANTIC_BATCH_SIZE" --num-workers "$SEMANTIC_NUM_WORKERS" \
    --epochs 200 --lr 3e-4 --candidate-count 32 --positive-count 4 \
    --top-k "$TOP_K" --retrieval-pool 64 --metadata-dim 64 \
    --gpu "$TRAIN_GPU" --seed "$SEED"
}

evaluate_semantic() {
  local split="$1"
  : "${SEMANTIC_CHECKPOINT:?Set SEMANTIC_CHECKPOINT to a semantic-stage best checkpoint}"
  "$ACTION_PY" semantic_main.py \
    --mode eval \
    --corpus "$CORPUS" \
    --split-path "$SEMANTIC_SPLIT" \
    --checkpoint "$SEMANTIC_CHECKPOINT" \
    --eval-split "$split" \
    --batch-size 128 \
    --num-workers "$SEMANTIC_NUM_WORKERS" \
    --top-k "$TOP_K" \
    --retrieval-pool 64 \
    --gpu "$TRAIN_GPU" \
    --seed "$SEED"
}

semantic_smoke() {
  [[ -f "$SEMANTIC_SPLIT" ]] || semantic_split
  mkdir -p "$SEMANTIC_RESULTS/smoke"
  "$ACTION_PY" semantic_main.py \
    --mode train-classifier \
    --corpus "$CORPUS" \
    --split-path "$SEMANTIC_SPLIT" \
    --output-dir "$SEMANTIC_RESULTS/smoke" \
    --run-name "semantic_smoke" \
    --batch-size 8 \
    --num-workers 0 \
    --epochs 1 \
    --candidate-count 8 \
    --positive-count 2 \
    --top-k 4 \
    --retrieval-pool 16 \
    --reference-batch-size 512 \
    --evidence-dim 256 \
    --encoder-hidden-dim 512 \
    --metadata-dim 64 \
    --dropout 0.1 \
    --limit-train-batches 2 \
    --limit-eval-batches 2 \
    --gpu "$TRAIN_GPU" \
    --seed "$SEED"
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
  : "${SEMANTIC_CHECKPOINT:?Set SEMANTIC_CHECKPOINT to the v2 GOE checkpoint}"
  mkdir -p "$CANDIDATES"
  "$ACTION_PY" scripts/precompute_action_candidates.py \
    --fs1000-root "$FS1000_ROOT" \
    --finefs-root "$FINEFS_ROOT" \
    --query-cache-dir "$FS1000_ROOT/static_dinov2_cls_patch_mean_cache" \
    --query-cache-prefix static_dinov2_cls_patch_mean \
    --corpus "$CORPUS" \
    --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
    --semantic-split "$SEMANTIC_SPLIT" \
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
  : "${SEMANTIC_CHECKPOINT:?Set SEMANTIC_CHECKPOINT to the v2 GOE checkpoint}"
  if [[ ! -f "$DYNAMIC_CHECKPOINT" ]]; then
    echo "Dynamic checkpoint not found: $DYNAMIC_CHECKPOINT" >&2
    return 2
  fi
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
    --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
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
  candidates-v2) precompute_candidates ;;
  semantic-split) semantic_split ;;
  semantic-smoke) semantic_smoke ;;
  train-semantic-classifier) train_semantic_classifier ;;
  train-semantic-goe) train_semantic_goe ;;
  semantic-val) evaluate_semantic val ;;
  semantic-test) evaluate_semantic test ;;
  preprocess-all)
    fs1000_times
    finefs_features
    build_corpus
    audit_retrieval
    ;;
  train-dynamic) train_dynamic ;;
  train-rag) train_rag ;;
  evaluate) evaluate ;;
  evaluate-baseline) evaluate_baseline ;;
  evaluate-dynamic) evaluate_baseline ;;
  *) usage; exit 2 ;;
esac
