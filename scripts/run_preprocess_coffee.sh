#!/bin/bash
set -e

# ============ CONFIG ============
H5="/home/ubuntu/ellina/data/mimicgen/coffee_d0/core/coffee_d0.hdf5"
VOX_CACHE="/home/ubuntu/ellina/data/mimicgen/coffee_d0/core/voxel_cache"
DLP_CFG="/home/ubuntu/ellina/lpwm-dev/logs/250126_061436_mimicgen_rgb_coffee_debug_more_kp/hparams.json"
DLP_CKPT="/home/ubuntu/ellina/lpwm-dev/logs/250126_061436_mimicgen_rgb_coffee_debug_more_kp/saves/last.pt"
OUT_PKL="/home/ubuntu/ellina/data/mimicgen/coffee_d0/core/200_traj_with_gripper_and_bg.pkl"
MAX_DEMOS=200
BATCH=4
NUM_GPUS=8
# ================================

cd /home/ubuntu/ellina/lpwm-dev

echo "Killing any existing processes..."
pkill -f ec_diffuser_voxel_preprocess 2>/dev/null || true
sleep 2

echo ""
echo "Launching $NUM_GPUS processes..."
echo ""

PIDS=()
for rank in $(seq 0 $((NUM_GPUS-1))); do
  OUT_SHARD="${OUT_PKL%.pkl}_rank${rank}.pkl"
  LOG_FILE="${OUT_PKL%.pkl}_rank${rank}.log"

  CUDA_VISIBLE_DEVICES=$rank \
  PYTHONPATH=. \
  python -u scripts/ec_diffuser_voxel_preprocess.py \
    --h5 "$H5" \
    --voxel-cache-dir "$VOX_CACHE" \
    --dlp-cfg "$DLP_CFG" \
    --dlp-ckpt "$DLP_CKPT" \
    --out-pkl "$OUT_SHARD" \
    --action-mode relative \
    --batch $BATCH \
    --max-demos $MAX_DEMOS \
    --rank $rank \
    --world-size $NUM_GPUS \
    > "$LOG_FILE" 2>&1 &

  PIDS+=($!)
  echo "  [Rank $rank] PID ${PIDS[$rank]} -> GPU $rank"
done

echo ""
echo "All processes launched. Monitoring progress..."
echo "Press Ctrl+C to stop monitoring (processes will continue)"
echo ""

# Monitor progress by tailing all logs
tail -f "${OUT_PKL%.pkl}"_rank*.log &
TAIL_PID=$!

# Wait for all processes
FAILED=0
for rank in $(seq 0 $((NUM_GPUS-1))); do
  if wait ${PIDS[$rank]}; then
    echo "[Rank $rank] Completed successfully"
  else
    echo "[Rank $rank] FAILED"
    FAILED=1
  fi
done

kill $TAIL_PID 2>/dev/null || true

if [ $FAILED -eq 1 ]; then
  echo ""
  echo "ERROR: One or more processes failed. Check logs."
  exit 1
fi

echo ""
echo "All processes completed. Merging shards..."
echo ""

PYTHONPATH=. python scripts/merge_preprocess_shards.py \
  --input-pattern "${OUT_PKL%.pkl}_rank*.pkl" \
  --output "$OUT_PKL" \
  --delete-shards

echo ""
echo "Done! Output: $OUT_PKL"
