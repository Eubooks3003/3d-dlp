#!/bin/bash
set -e

# ============================================================
#  RLBench end-to-end demo gen + voxelization (no PNGs, no PLYs)
# ============================================================
#
#  Spawns RLBench in-process (xvfb-run for headless machines),
#  rolls out fresh demos, fuses multi-camera RGBD in memory,
#  and writes voxel caches to per-episode voxel_cache/. No
#  RGB/depth/mask PNGs and no fused PLY files are written.
#
#  Usage:
#    bash scripts/gen_and_voxelize_rlbench.sh
#
#  Overrides (all env vars):
#    TASKS="close_jar open_drawer" bash scripts/gen_and_voxelize_rlbench.sh
#    TRAIN_EPISODES=200 bash scripts/gen_and_voxelize_rlbench.sh
#    START_EPISODE_IDX=21 bash scripts/gen_and_voxelize_rlbench.sh
#    SPLIT=train_data IMAGE_SIZE=128 bash scripts/gen_and_voxelize_rlbench.sh
#
#  Debug mode (visualize a few frames/voxels/kmeans per task on wandb,
#  no disk writes):
#    DEBUG=1 bash scripts/gen_and_voxelize_rlbench.sh
#    DEBUG=1 KMEANS=1 bash scripts/gen_and_voxelize_rlbench.sh
#    DEBUG=1 DEBUG_FRAMES=5 TASKS="close_jar open_drawer" \
#        bash scripts/gen_and_voxelize_rlbench.sh
#    DEBUG=1 WANDB_PROJECT=my-debug bash scripts/gen_and_voxelize_rlbench.sh
# ============================================================

# ============ CONFIGURE THESE ============
RLBENCH_ROOT="${RLBENCH_ROOT:-/home/ellina/Desktop/ManiGaussian/data}"
LPWM_DIR="${LPWM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MANIGAUSSIAN_DIR="${MANIGAUSSIAN_DIR:-/home/ellina/Desktop/ManiGaussian}"

# The 10 recommended RLBench tasks (from ManiGaussian)
DEFAULT_TASKS="close_jar open_drawer sweep_to_dustpan_of_size meat_off_grill turn_tap slide_block_to_color_target put_item_in_drawer reach_and_drag push_buttons stack_blocks"

SPLIT="${SPLIT:-train_data}"
CAMERAS="${CAMERAS:-front overhead left_shoulder right_shoulder}"
GRID_SIZE="${GRID_SIZE:-64 64 64}"

# Number of fresh episodes to generate per task.
TRAIN_EPISODES="${TRAIN_EPISODES:-100}"

IMAGE_SIZE="${IMAGE_SIZE:-128}"
MAX_POINTS="${MAX_POINTS:-200000}"
RENDERER="${RENDERER:-opengl}"

# Optional: override the starting episode index. By default, the Python
# script auto-detects the highest existing episodeN in the split/task and
# continues from the next index, so it's safe to re-run.
START_EPISODE_IDX="${START_EPISODE_IDX:-}"

# Optional: also precompute per-frame kmeans keypoints.
KMEANS="${KMEANS:-0}"
N_KP="${N_KP:-64}"

# Optional debug mode: roll out one demo per task and log a handful of frames
# (voxels + optional kmeans keypoints) to wandb. No voxel_cache / pkl files
# are written to disk. Use this for quick visual sanity-checks.
DEBUG="${DEBUG:-0}"
DEBUG_FRAMES="${DEBUG_FRAMES:-3}"
WANDB_PROJECT="${WANDB_PROJECT:-rlbench-gen-voxel-debug}"
# =========================================

if [[ -n "${TASKS:-}" ]]; then
    read -ra TASKS <<< "$TASKS"
else
    read -ra TASKS <<< "$DEFAULT_TASKS"
fi

echo "========================================"
echo " RLBench end-to-end voxel pipeline"
echo "========================================"
echo " Root:       $RLBENCH_ROOT"
echo " Split:      $SPLIT"
echo " Tasks:      ${TASKS[*]}"
echo " Cameras:    $CAMERAS"
echo " Grid:       $GRID_SIZE"
echo " Episodes:   $TRAIN_EPISODES per task"
echo " Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo " Renderer:   $RENDERER"
if [[ -n "$START_EPISODE_IDX" ]]; then
    echo " Start idx:  $START_EPISODE_IDX"
else
    echo " Start idx:  auto (continue after last existing episode)"
fi
if [[ "$DEBUG" == "1" ]]; then
    echo " DEBUG:      on ($DEBUG_FRAMES frames/task → wandb:$WANDB_PROJECT)"
fi
echo "========================================"
echo ""

if [[ ! -d "$MANIGAUSSIAN_DIR/third_party/RLBench" ]]; then
    echo "ERROR: RLBench not found at $MANIGAUSSIAN_DIR/third_party/RLBench"
    echo "  Set MANIGAUSSIAN_DIR to your ManiGaussian checkout."
    exit 1
fi

cd "$LPWM_DIR"

# RLBench / PyRep live under ManiGaussian/third_party so we don't force the
# user to pip-install them system-wide.
RLBENCH_PY="$MANIGAUSSIAN_DIR/third_party/RLBench"
PYREP_PY="$MANIGAUSSIAN_DIR/third_party/PyRep"
export PYTHONPATH="$LPWM_DIR:$RLBENCH_PY:$PYREP_PY:${PYTHONPATH:-}"

CMD=(python -u scripts/gen_and_voxelize_rlbench.py
    --root "$RLBENCH_ROOT"
    --split "$SPLIT"
    --tasks ${TASKS[*]}
    --episodes-per-task "$TRAIN_EPISODES"
    --image-size "$IMAGE_SIZE"
    --cameras $CAMERAS
    --grid-whd $GRID_SIZE
    --max-fuse-points "$MAX_POINTS"
    --rgbo
    --crop-workspace
    --renderer "$RENDERER"
)

if [[ -n "$START_EPISODE_IDX" ]]; then
    CMD+=(--start-episode-idx "$START_EPISODE_IDX")
fi
if [[ "$KMEANS" == "1" ]]; then
    CMD+=(--kmeans --n-kp "$N_KP")
fi
if [[ "$DEBUG" == "1" ]]; then
    CMD+=(--debug --debug-frames "$DEBUG_FRAMES" --wandb-project "$WANDB_PROJECT")
fi

xvfb-run -a "${CMD[@]}"

echo ""
echo "========================================"
echo " Done."
echo "========================================"
echo " Voxel caches written to:"
echo "   $RLBENCH_ROOT/$SPLIT/{task}/all_variations/episodes/episodeN/voxel_cache/"
echo ""
