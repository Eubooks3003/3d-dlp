#!/bin/bash
set -e

# ============ CONFIGURE THESE ============
DATA_ROOT="/home/ellina/Desktop/data/3D-DLP-mimicgen-data"
LPWM_DIR="/home/ellina/Desktop/lpwm-copy"
DLP_CFG=""   # e.g. /path/to/hparams.json
DLP_CKPT=""  # e.g. /path/to/saves/last.pt
OUT_DIR="${DATA_ROOT}/preprocessed"

BATCH=8
ACTION_MODE="relative"
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
# =========================================

if [[ -z "$DLP_CFG" || -z "$DLP_CKPT" ]]; then
    echo "ERROR: Set DLP_CFG and DLP_CKPT in this script before running."
    exit 1
fi

TASKS=(
    coffee_d0
    coffee_preparation_d0
    hammer_cleanup_d0
    kitchen_d0
    mug_cleanup_d0
    nut_assembly_d0
    pick_place_d0
    square_d0
    stack_d0
    stack_three_d0
    threading_d0
    three_piece_assembly_d0
)

mkdir -p "$OUT_DIR"
cd "$LPWM_DIR"

for TASK in "${TASKS[@]}"; do
    H5="${DATA_ROOT}/core/${TASK}.hdf5"
    VOX_CACHE="${DATA_ROOT}/${TASK}/voxel_cache/voxel"
    OUT_PKL="${OUT_DIR}/${TASK}.pkl"

    if [[ ! -f "$H5" ]]; then
        echo "[SKIP] $TASK: H5 not found at $H5"
        continue
    fi
    if [[ ! -d "$VOX_CACHE" ]]; then
        echo "[SKIP] $TASK: voxel cache not found at $VOX_CACHE"
        continue
    fi
    if [[ -f "$OUT_PKL" ]]; then
        echo "[SKIP] $TASK: output already exists at $OUT_PKL"
        continue
    fi

    echo ""
    echo "========================================"
    echo "  Processing: $TASK"
    echo "========================================"

    if [[ "$NUM_GPUS" -gt 1 ]]; then
        # Multi-GPU: launch one process per GPU, then merge
        PIDS=()
        for ((rank=0; rank<NUM_GPUS; rank++)); do
            LOG_FILE="${OUT_DIR}/${TASK}_rank${rank}.log"
            CUDA_VISIBLE_DEVICES=$rank \
            PYTHONPATH=. \
            python -u scripts/ec_diffuser_voxel_preprocess.py \
                --h5 "$H5" \
                --voxel-cache-dir "$VOX_CACHE" \
                --dlp-cfg "$DLP_CFG" \
                --dlp-ckpt "$DLP_CKPT" \
                --out-pkl "$OUT_PKL" \
                --action-mode "$ACTION_MODE" \
                --batch "$BATCH" \
                --rank "$rank" \
                --world-size "$NUM_GPUS" \
                > "$LOG_FILE" 2>&1 &
            PIDS+=($!)
            echo "  [Rank $rank] PID $! -> GPU $rank"
        done

        FAILED=0
        for ((rank=0; rank<NUM_GPUS; rank++)); do
            if wait ${PIDS[$rank]}; then
                echo "  [Rank $rank] done"
            else
                echo "  [Rank $rank] FAILED (see ${OUT_DIR}/${TASK}_rank${rank}.log)"
                FAILED=1
            fi
        done

        if [[ $FAILED -eq 1 ]]; then
            echo "[ERROR] $TASK failed, skipping merge"
            continue
        fi

        # Merge shards
        PYTHONPATH=. python scripts/merge_preprocess_shards.py \
            --input-pattern "${OUT_PKL%.pkl}_rank*.pkl" \
            --output "$OUT_PKL" \
            --delete-shards

    else
        # Single GPU
        PYTHONPATH=. \
        python -u scripts/ec_diffuser_voxel_preprocess.py \
            --h5 "$H5" \
            --voxel-cache-dir "$VOX_CACHE" \
            --dlp-cfg "$DLP_CFG" \
            --dlp-ckpt "$DLP_CKPT" \
            --out-pkl "$OUT_PKL" \
            --action-mode "$ACTION_MODE" \
            --batch "$BATCH" \
            --device cuda
    fi

    echo "[DONE] $TASK -> $OUT_PKL"
done

echo ""
echo "========================================"
echo "  All tasks complete!"
echo "  Output: $OUT_DIR/"
echo "========================================"
