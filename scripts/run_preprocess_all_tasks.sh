#!/bin/bash
set -e

# ============ CONFIGURE THESE ============
DATA_ROOT="/home/ubuntu/tal-lpwm-neurips-2026/data/3D-DLP-mimicgen-data"
LPWM_DIR="/home/ubuntu/tal-lpwm-neurips-2026/ellina/lpwm-dev"
DLP_CFG="${LPWM_DIR}/logs/010326_173829_mimicgen_multitask/hparams.json"
DLP_CKPT="${LPWM_DIR}/logs/010326_173829_mimicgen_multitask/saves/best.pt"
OUT_DIR="${DATA_ROOT}/preprocessed"

BATCH=8
ACTION_MODE="relative"
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
# =========================================

# ---------- Debug mode ----------
# Usage:  bash run_preprocess_all_tasks.sh --debug [NUM_SAMPLES] [WANDB_PROJECT]
DEBUG=0
DEBUG_SAMPLES=3
WANDB_PROJECT="ec-diffuser-debug"

if [[ "$1" == "--debug" ]]; then
    DEBUG=1
    if [[ -n "$2" ]]; then DEBUG_SAMPLES="$2"; fi
    if [[ -n "$3" ]]; then WANDB_PROJECT="$3"; fi
fi
# --------------------------------

if [[ ! -f "$DLP_CFG" ]]; then
    echo "ERROR: DLP_CFG not found at $DLP_CFG"
    echo "  If hparams.json is elsewhere, update DLP_CFG in this script."
    exit 1
fi
if [[ ! -f "$DLP_CKPT" ]]; then
    echo "ERROR: DLP_CKPT not found at $DLP_CKPT"
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

# ---------- Debug: run on first available task then exit ----------
if [[ "$DEBUG" -eq 1 ]]; then
    echo "========================================"
    echo "  DEBUG MODE — encode/decode sanity check"
    echo "  Samples: $DEBUG_SAMPLES | wandb project: $WANDB_PROJECT"
    echo "========================================"

    for TASK in "${TASKS[@]}"; do
        H5="${DATA_ROOT}/core/${TASK}.hdf5"
        VOX_CACHE="${DATA_ROOT}/${TASK}/voxel_cache/voxel"

        if [[ ! -f "$H5" ]]; then continue; fi
        if [[ ! -d "$VOX_CACHE" ]]; then continue; fi

        echo "  Using task: $TASK"
        PYTHONPATH=. \
        python -u scripts/ec_diffuser_voxel_preprocess.py \
            --h5 "$H5" \
            --voxel-cache-dir "$VOX_CACHE" \
            --dlp-cfg "$DLP_CFG" \
            --dlp-ckpt "$DLP_CKPT" \
            --out-pkl "/dev/null" \
            --action-mode "$ACTION_MODE" \
            --batch "$BATCH" \
            --device cuda \
            --debug \
            --debug-samples "$DEBUG_SAMPLES" \
            --wandb-project "$WANDB_PROJECT"

        echo "[DEBUG DONE] Check wandb project '$WANDB_PROJECT'"
        exit 0
    done

    echo "ERROR: No task found with both H5 and voxel cache."
    exit 1
fi

# ==========================================================
#  STEP 1: Precompute kmeans for ALL tasks (multi-worker)
# ==========================================================
echo ""
echo "========================================"
echo "  Step 1: Precompute kmeans (all tasks)"
echo "  ${WORKERS_PER_GPU} workers x ${NUM_GPUS} GPU(s)"
echo "========================================"

PYTHONPATH=. python -u scripts/precompute_kmeans.py \
    --data-root "$DATA_ROOT" \
    --dlp-cfg "$DLP_CFG" \
    --dlp-ckpt "$DLP_CKPT" \
    --num-gpus "$NUM_GPUS" \
    --workers-per-gpu "$WORKERS_PER_GPU"

echo "[Step 1 DONE] KMeans cache ready"
echo ""

# ==========================================================
#  STEP 2: Encode tokens (kmeans cached, GPU goes fast now)
# ==========================================================
echo "========================================"
echo "  Step 2: Encode tokens (all tasks)"
echo "========================================"

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
