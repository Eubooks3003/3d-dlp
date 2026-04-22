#!/bin/bash
# Preprocess all 10 RLBench tasks for EC-Diffuser (3D voxel DLP).
#
# Mirrors lpwm-copy/scripts/run_preprocess_rlbench_all_tasks.sh (the 2D DLP
# version), but consumes the per-episode voxel_cache/ + kmeans_cache/ and
# encodes them with the 3D voxel DLP instead of RGB PNGs.
#
# Prerequisites on the remote (already present for the user's data):
#   <ROOT>/<SPLIT>/<task>/all_variations/episodes/episode<N>/
#       low_dim_obs.pkl
#       variation_descriptions.pkl
#       variation_number.pkl
#       voxel_cache/NNNNNN_voxels.pt       # from preprocess_rlbench_rgbo_all.sh
#                                          #   or gen_and_voxelize_rlbench.sh
#       kmeans_cache/NNNNNN_kmeans.pt      # from precompute_kmeans_rlbench.py
#                                          #   or gen_and_voxelize_rlbench.sh KMEANS=1
#
# If a few kmeans files are missing, the Python encoder will backfill them
# sequentially. For a cold cache, run precompute_kmeans_rlbench.py first.
#
# Usage (defaults set for the remote layout):
#   bash scripts/run_preprocess_rlbench_voxel_all_tasks.sh
#
# Overrides via env vars:
#   TASKS="close_jar open_drawer"  bash scripts/run_preprocess_rlbench_voxel_all_tasks.sh
#   SPLIT=test_data                bash scripts/run_preprocess_rlbench_voxel_all_tasks.sh
#   MAX_DEMOS=20                   bash scripts/run_preprocess_rlbench_voxel_all_tasks.sh
#   NUM_GPUS=4                     bash scripts/run_preprocess_rlbench_voxel_all_tasks.sh
#   BATCH=16                       bash scripts/run_preprocess_rlbench_voxel_all_tasks.sh
#   OUT_DIR=/some/where            bash scripts/run_preprocess_rlbench_voxel_all_tasks.sh
#
# Debug mode (encode/decode a few frames per task, log to wandb, no pickles):
#   bash scripts/run_preprocess_rlbench_voxel_all_tasks.sh --debug
#   bash scripts/run_preprocess_rlbench_voxel_all_tasks.sh --debug 5
#   bash scripts/run_preprocess_rlbench_voxel_all_tasks.sh --debug 5 my-wandb-project

set -e

# ---------- Debug flag parsing ----------
DEBUG=0
DEBUG_SAMPLES=3
WANDB_PROJECT="rlbench-dlp-debug"
if [[ "${1:-}" == "--debug" ]]; then
    DEBUG=1
    if [[ -n "${2:-}" ]]; then DEBUG_SAMPLES="$2"; fi
    if [[ -n "${3:-}" ]]; then WANDB_PROJECT="$3"; fi
fi
# ----------------------------------------

ROOT="${ROOT:-/home/ubuntu/tal-lpwm-neurips-2026/data/rlbench}"
SPLIT="${SPLIT:-train_data}"
LPWM_DIR="${LPWM_DIR:-/home/ubuntu/tal-lpwm-neurips-2026/ellina/lpwm-dev}"
DLP_CFG="${DLP_CFG:-${LPWM_DIR}/logs/180426_223153_rlbench_multitask/hparams.json}"
DLP_CKPT="${DLP_CKPT:-${LPWM_DIR}/logs/180426_223153_rlbench_multitask/saves/best.pt}"
OUT_DIR="${OUT_DIR:-${ROOT}/preprocessed_voxel_tokens}"
BATCH="${BATCH:-8}"
MAX_DEMOS="${MAX_DEMOS:-}"   # blank = all episodes
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
: "${NUM_GPUS:=1}"

DEFAULT_TASKS="close_jar open_drawer sweep_to_dustpan_of_size meat_off_grill turn_tap slide_block_to_color_target put_item_in_drawer reach_and_drag push_buttons stack_blocks"
if [[ -z "${TASKS:-}" ]]; then
    TASKS="$DEFAULT_TASKS"
fi

echo "============================================================"
echo "  RLBench voxel DLP preprocessing"
echo "============================================================"
echo "ROOT      = $ROOT"
echo "SPLIT     = $SPLIT"
echo "OUT_DIR   = $OUT_DIR"
echo "DLP_CFG   = $DLP_CFG"
echo "DLP_CKPT  = $DLP_CKPT"
echo "BATCH     = $BATCH"
echo "NUM_GPUS  = $NUM_GPUS"
echo "MAX_DEMOS = ${MAX_DEMOS:-<all>}"
echo "TASKS     = $TASKS"
echo "============================================================"
echo ""

if [[ ! -f "$DLP_CFG" ]]; then
    echo "ERROR: DLP_CFG not found at $DLP_CFG"
    exit 1
fi
if [[ ! -f "$DLP_CKPT" ]]; then
    echo "ERROR: DLP_CKPT not found at $DLP_CKPT"
    exit 1
fi
if [[ ! -d "$LPWM_DIR" ]]; then
    echo "ERROR: LPWM_DIR not found at $LPWM_DIR"
    exit 1
fi

cd "$LPWM_DIR"

# ---------- Debug: encode/decode N frames per task, log to wandb, exit ----------
if [[ "$DEBUG" -eq 1 ]]; then
    echo "========================================"
    echo "  DEBUG MODE — encode/decode sanity check"
    echo "  Samples/task: $DEBUG_SAMPLES | wandb project: $WANDB_PROJECT"
    echo "========================================"

    PYTHONPATH=. python -u scripts/ec_diffuser_voxel_preprocess_rlbench.py \
        --root "$ROOT" \
        --split "$SPLIT" \
        --dlp-cfg  "$DLP_CFG" \
        --dlp-ckpt "$DLP_CKPT" \
        --device cuda \
        --debug \
        --debug-samples "$DEBUG_SAMPLES" \
        --wandb-project "$WANDB_PROJECT" \
        --debug-tasks $TASKS

    echo ""
    echo "[DEBUG DONE] Check wandb project '$WANDB_PROJECT'"
    exit 0
fi
# --------------------------------------------------------------------------------

mkdir -p "$OUT_DIR"

FAILED=()
for task in $TASKS; do
    TASK_DIR="$OUT_DIR/rlbench_${task}"
    OUT_PKL="$TASK_DIR/${task}.pkl"

    if [[ -f "$OUT_PKL" ]]; then
        echo "[SKIP] $task: output already exists at $OUT_PKL"
        continue
    fi

    mkdir -p "$TASK_DIR"

    # Snapshot DLP cfg + ckpt next to the pickle so we know exactly which
    # weights produced it (best.pt may be overwritten by later training).
    cp "$DLP_CFG"  "$TASK_DIR/dlp_config.json"
    cp "$DLP_CKPT" "$TASK_DIR/dlp_ckpt.pt"

    echo ""
    echo "### [$task] $(date +%H:%M:%S)"

    ARGS=(--root "$ROOT"
          --split "$SPLIT"
          --task "$task"
          --dlp-cfg  "$TASK_DIR/dlp_config.json"
          --dlp-ckpt "$TASK_DIR/dlp_ckpt.pt"
          --out-pkl  "$OUT_PKL"
          --batch    "$BATCH")
    if [[ -n "$MAX_DEMOS" ]]; then
        ARGS+=(--max-demos "$MAX_DEMOS")
    fi

    if [[ "$NUM_GPUS" -gt 1 ]]; then
        # Multi-GPU: one process per GPU, then merge shards.
        PIDS=()
        for ((rank=0; rank<NUM_GPUS; rank++)); do
            LOG_FILE="$TASK_DIR/${task}_rank${rank}.log"
            CUDA_VISIBLE_DEVICES=$rank PYTHONPATH=. \
                python -u scripts/ec_diffuser_voxel_preprocess_rlbench.py \
                    "${ARGS[@]}" \
                    --rank "$rank" \
                    --world-size "$NUM_GPUS" \
                    --device cuda \
                    > "$LOG_FILE" 2>&1 &
            PIDS+=($!)
            echo "  [rank $rank] PID $! -> GPU $rank (log: $LOG_FILE)"
        done

        RANK_FAILED=0
        for ((rank=0; rank<NUM_GPUS; rank++)); do
            if wait ${PIDS[$rank]}; then
                echo "  [rank $rank] done"
            else
                echo "  [rank $rank] FAILED (see $TASK_DIR/${task}_rank${rank}.log)"
                RANK_FAILED=1
            fi
        done

        if [[ $RANK_FAILED -eq 1 ]]; then
            echo "!!! [$task] FAILED — continuing"
            FAILED+=("$task")
            continue
        fi

        PYTHONPATH=. python scripts/merge_preprocess_shards.py \
            --input-pattern "${OUT_PKL%.pkl}_rank*.pkl" \
            --output "$OUT_PKL" \
            --delete-shards
    else
        if ! PYTHONPATH=. python -u scripts/ec_diffuser_voxel_preprocess_rlbench.py \
                "${ARGS[@]}" --device cuda; then
            echo "!!! [$task] FAILED — continuing"
            FAILED+=("$task")
            continue
        fi
    fi

    echo "[DONE] $task -> $OUT_PKL"
done

echo ""
echo "============================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "  All tasks preprocessed successfully."
    echo "  Output: $OUT_DIR/rlbench_<task>/<task>.pkl"
else
    echo "  FAILED tasks: ${FAILED[*]}"
    exit 1
fi
echo "============================================================"
