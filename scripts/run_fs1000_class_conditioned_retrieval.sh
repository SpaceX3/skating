#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-train}"
REPO_DIR="${REPO_DIR:-/home/v100/.worktrees/skating-videomae-static}"
DATASET_ROOT="${DATASET_ROOT:-/home/v100/ZYQ/FS1000 Dataset}"
FS1000_FEATURE_ROOT="${FS1000_FEATURE_ROOT:-/media/v100/disk3t/finefs_pocr_classifier/features/fs1000_videomae_base_1s_stride05}"
FINEFS_FEATURE_ROOT="${FINEFS_FEATURE_ROOT:-/media/v100/disk3t/finefs_pocr_classifier/features/videomae_base_1s_stride05}"
FINEFS_MANIFEST="${FINEFS_MANIFEST:-/media/v100/disk3t/finefs_pocr_classifier/experiments/e11_video_grain_split/cliplet_manifest.jsonl}"
BANK_DIR="${BANK_DIR:-/media/v100/disk3t/skating/finefs_c1_class_bank_first_token}"
CACHE_DIR="${CACHE_DIR:-/media/v100/disk3t/skating/fs1000_static_videomae_c1_class_retrieval}"
RUN_DIR="${RUN_DIR:-/media/v100/disk3t/skating/experiments/videomae_c1_class_retrieval/manual_seed2026}"
GPU_UUID="${GPU_UUID:-GPU-f40ff723-535e-10df-74d9-4b38ebeac3c5}"
SKATING_PYTHON="${SKATING_PYTHON:-/home/v100/anaconda3/envs/skating-action/bin/python}"
C1_PYTHON="${C1_PYTHON:-/home/v100/anaconda3/envs/skating-action-e10/bin/python}"
INIT_DYNAMIC_CHECKPOINT="${INIT_DYNAMIC_CHECKPOINT:-/home/v100/ZYQ/skating/fs800_result/checkpoint_best_0.872.pth}"

cd "$REPO_DIR"

run_bank() {
  "$SKATING_PYTHON" build_finefs_class_bank.py \
    --manifest "$FINEFS_MANIFEST" \
    --feature-root "$FINEFS_FEATURE_ROOT" \
    --output-dir "$BANK_DIR"
}

run_cache() {
  CUDA_VISIBLE_DEVICES="$GPU_UUID" "$C1_PYTHON" precompute_class_conditioned_static.py \
    --dataset-root "$DATASET_ROOT" \
    --feature-root "$FS1000_FEATURE_ROOT" \
    --bank-dir "$BANK_DIR" \
    --output-dir "$CACHE_DIR" \
    --device cuda:0 \
    --top-classes 2 \
    --top-k 4 \
    --temperature 0.1 \
    --probability-power 1.0
}

run_train() {
  CUDA_VISIBLE_DEVICES="$GPU_UUID" "$SKATING_PYTHON" main.py \
    --gpu 0 \
    --root-path "$DATASET_ROOT" \
    --static-cache-dir "$CACHE_DIR" \
    --static-cache-prefix static_videomae_c1_class_retrieval \
    --static-feature-dim 1536 \
    --init-dynamic-checkpoint "$INIT_DYNAMIC_CHECKPOINT" \
    --seed 2026 \
    --epochs 200 \
    --batch-size 16 \
    --num-workers 8 \
    --log-dir "$RUN_DIR"
}

run_eval() {
  for checkpoint in best_spearman.pth best_loss.pth; do
    CUDA_VISIBLE_DEVICES="$GPU_UUID" "$SKATING_PYTHON" eval.py \
      --gpu 0 \
      --root-path "$DATASET_ROOT" \
      --static-cache-dir "$CACHE_DIR" \
      --static-cache-prefix static_videomae_c1_class_retrieval \
      --static-feature-dim 1536 \
      --checkpoint "$RUN_DIR/$checkpoint"
    echo
  done
}

case "$MODE" in
  bank) run_bank ;;
  cache) run_cache ;;
  train) run_train ;;
  eval) run_eval ;;
  all)
    run_bank
    run_cache
    run_train
    run_eval
    ;;
  *)
    echo "usage: $0 {bank|cache|train|eval|all}" >&2
    exit 2
    ;;
esac
