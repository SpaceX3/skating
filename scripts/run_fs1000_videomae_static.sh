#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
REPO_DIR="${REPO_DIR:-/home/v100/.worktrees/skating-videomae-static}"
DATASET_ROOT="${DATASET_ROOT:-/home/v100/ZYQ/FS1000 Dataset}"
FEATURE_ROOT="${FEATURE_ROOT:-/media/v100/disk3t/finefs_pocr_classifier/features/fs1000_videomae_base_1s_stride05}"
CACHE_DIR="${CACHE_DIR:-/media/v100/disk3t/skating/fs1000_static_videomae_c1}"
RUN_DIR="${RUN_DIR:-/media/v100/disk3t/skating/experiments/videomae_static_c1/manual_seed2026}"
GPU_UUID="${GPU_UUID:-GPU-f40ff723-535e-10df-74d9-4b38ebeac3c5}"
SKATING_PYTHON="${SKATING_PYTHON:-/home/v100/anaconda3/envs/skating-action/bin/python}"
C1_PYTHON="${C1_PYTHON:-/home/v100/anaconda3/envs/skating-action-e10/bin/python}"

cd "$REPO_DIR"

run_cache() {
  if [[ -f "$CACHE_DIR/cache_report.json" ]]; then
    echo "Reusing audited cache: $CACHE_DIR"
    return
  fi
  CUDA_VISIBLE_DEVICES="$GPU_UUID" "$C1_PYTHON" precompute_videomae_static.py \
    --dataset-root "$DATASET_ROOT" \
    --feature-root "$FEATURE_ROOT" \
    --output-dir "$CACHE_DIR" \
    --device cuda:0 \
    --batch-size 512
}

run_train() {
  CUDA_VISIBLE_DEVICES="$GPU_UUID" "$SKATING_PYTHON" main.py \
    --gpu 0 \
    --root-path "$DATASET_ROOT" \
    --static-cache-dir "$CACHE_DIR" \
    --static-cache-prefix static_videomae_c1 \
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
      --static-cache-prefix static_videomae_c1 \
      --checkpoint "$RUN_DIR/$checkpoint"
    echo
  done
}

case "$MODE" in
  cache)
    run_cache
    ;;
  train)
    run_train
    ;;
  eval)
    run_eval
    ;;
  all)
    run_cache
    run_train
    run_eval
    ;;
  *)
    echo "usage: $0 {cache|train|eval|all}" >&2
    exit 2
    ;;
esac
