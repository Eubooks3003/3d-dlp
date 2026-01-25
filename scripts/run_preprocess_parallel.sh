#!/bin/bash
# Parallel preprocessing across multiple GPUs
#
# Usage:
#   ./run_preprocess_parallel.sh --h5 /path/to/data.h5 \
#       --voxel-cache-dir /path/to/voxel_cache \
#       --dlp-cfg /path/to/config.json \
#       --dlp-ckpt /path/to/checkpoint.pt \
#       --out-pkl /path/to/output.pkl \
#       [--num-gpus 4] [--max-demos 100] [--batch 16]
#
# After completion, run merge_preprocess_shards.py to combine outputs.

set -e

# Default values
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
BATCH=16
MAX_DEMOS=""
ACTION_MODE="absolute"

# Parse arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --h5)
            H5_PATH="$2"
            shift 2
            ;;
        --voxel-cache-dir)
            VOXEL_CACHE="$2"
            shift 2
            ;;
        --dlp-cfg)
            DLP_CFG="$2"
            shift 2
            ;;
        --dlp-ckpt)
            DLP_CKPT="$2"
            shift 2
            ;;
        --out-pkl)
            OUT_PKL="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --max-demos)
            MAX_DEMOS="$2"
            shift 2
            ;;
        --action-mode)
            ACTION_MODE="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$H5_PATH" || -z "$VOXEL_CACHE" || -z "$DLP_CFG" || -z "$DLP_CKPT" || -z "$OUT_PKL" ]]; then
    echo "Error: Missing required arguments"
    echo "Required: --h5, --voxel-cache-dir, --dlp-cfg, --dlp-ckpt, --out-pkl"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROCESS_SCRIPT="$SCRIPT_DIR/ec_diffuser_voxel_preprocess.py"

echo "============================================"
echo "Parallel Preprocessing"
echo "============================================"
echo "GPUs: $NUM_GPUS"
echo "H5: $H5_PATH"
echo "Voxel Cache: $VOXEL_CACHE"
echo "DLP Config: $DLP_CFG"
echo "DLP Checkpoint: $DLP_CKPT"
echo "Output: $OUT_PKL"
echo "Batch size: $BATCH"
echo "Action mode: $ACTION_MODE"
[[ -n "$MAX_DEMOS" ]] && echo "Max demos: $MAX_DEMOS"
echo "============================================"

# Build base command
BASE_CMD="python $PREPROCESS_SCRIPT \
    --h5 $H5_PATH \
    --voxel-cache-dir $VOXEL_CACHE \
    --dlp-cfg $DLP_CFG \
    --dlp-ckpt $DLP_CKPT \
    --out-pkl $OUT_PKL \
    --batch $BATCH \
    --action-mode $ACTION_MODE \
    --world-size $NUM_GPUS"

[[ -n "$MAX_DEMOS" ]] && BASE_CMD="$BASE_CMD --max-demos $MAX_DEMOS"

# Launch processes in parallel
PIDS=()
for ((rank=0; rank<NUM_GPUS; rank++)); do
    echo "[Rank $rank] Launching on GPU $rank..."
    CUDA_VISIBLE_DEVICES=$rank $BASE_CMD --rank $rank --device cuda &
    PIDS+=($!)
done

# Wait for all processes
echo ""
echo "Waiting for all processes to complete..."
FAILED=0
for ((rank=0; rank<NUM_GPUS; rank++)); do
    if wait ${PIDS[$rank]}; then
        echo "[Rank $rank] Completed successfully"
    else
        echo "[Rank $rank] FAILED"
        FAILED=1
    fi
done

if [[ $FAILED -eq 1 ]]; then
    echo ""
    echo "ERROR: One or more processes failed!"
    exit 1
fi

echo ""
echo "============================================"
echo "All processes completed!"
echo "============================================"
echo ""
echo "Now merge the shards with:"
echo "  python $SCRIPT_DIR/merge_preprocess_shards.py \\"
echo "      --input-pattern '${OUT_PKL%.pkl}_rank*.pkl' \\"
echo "      --output $OUT_PKL"
