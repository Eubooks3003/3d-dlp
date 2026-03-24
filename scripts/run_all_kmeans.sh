#!/usr/bin/env bash
set -euo pipefail

# Precompute kmeans cache for all tasks.
# The DLP prior uses FeatureKMeansRGB (pure algorithmic k-means in
# Lab color + XYZ space) — no trained checkpoint needed.
#
# Usage:
#   cd /home/ellina/Desktop/lpwm-dev && bash scripts/run_all_kmeans.sh
#
# Override defaults with env vars:
#   TASKS="stack_d1 coffee_d2" BATCH=16 bash scripts/run_all_kmeans.sh

DATA_ROOT="/home/ellina/Desktop/data/3D-DLP-mimicgen-data"
DLP_CFG="/home/ellina/Desktop/lpwm-dev/configs/mimicgen_multitask.json"
WORKERS_PER_GPU=${WORKERS_PER_GPU:-1}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
BATCH=${BATCH:-4}

TASKS=${TASKS:-"stack_d1 stack_three_d1 square_d2 threading_d2 coffee_d2 three_piece_assembly_d2 hammer_cleanup_d1 mug_cleanup_d1 kitchen_d1 coffee_preparation_d1"}

echo "========================================"
echo "  Precompute kmeans (rgb_km prior)"
echo "  Tasks: $TASKS"
echo "  ${WORKERS_PER_GPU} workers x ${NUM_GPUS} GPU(s), batch ${BATCH}"
echo "========================================"

cd /home/ellina/Desktop/lpwm-dev

PYTHONPATH=. python -u scripts/precompute_kmeans.py \
    --data-root "$DATA_ROOT" \
    --dlp-cfg "$DLP_CFG" \
    --num-gpus "$NUM_GPUS" \
    --workers-per-gpu "$WORKERS_PER_GPU" \
    --batch "$BATCH" \
    --tasks $TASKS

echo ""
echo "[DONE] KMeans cache ready for all tasks"
