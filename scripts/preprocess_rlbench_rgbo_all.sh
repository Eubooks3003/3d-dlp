#!/bin/bash
set -e

# ============================================================
#  RLBench RGBO Preprocessing Pipeline
# ============================================================
#
#  Full pipeline for generating 4-channel (RGB + Occupancy)
#  voxel data from RLBench demonstrations.
#
#  Steps:
#    0. Generate RLBench demonstrations  (optional, skip with SKIP_GEN_DEMOS=1)
#    1. Fuse multi-camera RGB+depth into PLY point clouds
#    2. Voxelize PLY files into RGBO (4-channel) voxels
#
#  Usage:
#    bash scripts/preprocess_rlbench_rgbo_all.sh
#
#  Override tasks:
#    TASKS="close_jar open_drawer" bash scripts/preprocess_rlbench_rgbo_all.sh
#
#  Skip demo generation (if you already have RLBench data):
#    SKIP_GEN_DEMOS=1 bash scripts/preprocess_rlbench_rgbo_all.sh
#
#  Process only specific splits:
#    SPLITS="train" bash scripts/preprocess_rlbench_rgbo_all.sh
#
#  Debug mode (voxelize one frame per task, log to wandb):
#    bash scripts/preprocess_rlbench_rgbo_all.sh --debug
#
# ============================================================

# ============ CONFIGURE THESE ============
RLBENCH_ROOT="${RLBENCH_ROOT:-/home/ellina/Desktop/ManiGaussian/data}"
LPWM_DIR="${LPWM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MANIGAUSSIAN_DIR="${MANIGAUSSIAN_DIR:-/home/ellina/Desktop/ManiGaussian}"

# The 10 recommended RLBench tasks (from ManiGaussian)
DEFAULT_TASKS="close_jar open_drawer sweep_to_dustpan_of_size meat_off_grill turn_tap slide_block_to_color_target put_item_in_drawer reach_and_drag push_buttons stack_blocks"

# Splits to process
SPLITS="${SPLITS:-train_data test_data}"

# Camera views to fuse into point clouds
CAMERAS="${CAMERAS:-front overhead left_shoulder right_shoulder}"

# Voxel grid resolution
GRID_SIZE="${GRID_SIZE:-64 64 64}"

# Depth image scale (1000 = millimeters, 1 = meters)
DEPTH_SCALE="${DEPTH_SCALE:-1000.0}"

# Max points per fused point cloud
MAX_POINTS="${MAX_POINTS:-200000}"

# Camera field of view (degrees)
FOV="${FOV:-60.0}"

# Number of episodes per task
TRAIN_EPISODES="${TRAIN_EPISODES:-20}"
TEST_EPISODES="${TEST_EPISODES:-5}"

# Image size for demo generation
IMAGE_SIZE="${IMAGE_SIZE:-128}"

# Wandb project for debug mode
WANDB_PROJECT="${WANDB_PROJECT:-rlbench-voxel-debug}"
# =========================================

# ---------- Debug mode ----------
# Usage:  bash scripts/preprocess_rlbench_rgbo_all.sh --debug [WANDB_PROJECT]
DEBUG=0
if [[ "${1:-}" == "--debug" ]]; then
    DEBUG=1
    if [[ -n "${2:-}" ]]; then WANDB_PROJECT="$2"; fi
fi
# --------------------------------

# Parse task list
if [[ -n "${TASKS:-}" ]]; then
    read -ra TASKS <<< "$TASKS"
else
    read -ra TASKS <<< "$DEFAULT_TASKS"
fi

read -ra SPLIT_ARR <<< "$SPLITS"

echo "========================================"
echo "  RLBench RGBO Preprocessing Pipeline"
echo "========================================"
echo "Root:     $RLBENCH_ROOT"
echo "Tasks:    ${TASKS[*]}"
echo "Splits:   $SPLITS"
echo "Cameras:  $CAMERAS"
echo "Grid:     $GRID_SIZE"
echo "LPWM dir: $LPWM_DIR"
echo "========================================"
echo ""

# ==========================================================
#  DEBUG MODE: voxelize one frame per task, log to wandb
# ==========================================================
if [[ "$DEBUG" -eq 1 ]]; then
    echo "========================================"
    echo "  DEBUG MODE — one frame per task -> wandb"
    echo "  wandb project: $WANDB_PROJECT"
    echo "========================================"

    cd "$LPWM_DIR"

    PYTHONPATH=. python -u scripts/preprocess_rlbench_voxels.py \
        --root "$RLBENCH_ROOT" \
        --splits ${SPLIT_ARR[*]} \
        --tasks ${TASKS[*]} \
        --grid_whd $GRID_SIZE \
        --voxel_mode avg_rgb \
        --rgbo \
        --cameras $CAMERAS \
        --depth_scale "$DEPTH_SCALE" \
        --fov "$FOV" \
        --max_fuse_points "$MAX_POINTS" \
        --crop_workspace \
        --kmeans --n_kp 64 \
        --debug \
        --wandb_project "$WANDB_PROJECT"

    echo ""
    echo "[DEBUG DONE] Check wandb project '$WANDB_PROJECT'"
    exit 0
fi

# ==========================================================
#  STEP 0: Generate RLBench demonstrations (optional)
# ==========================================================
if [[ "${SKIP_GEN_DEMOS:-0}" != "1" ]]; then
    echo "========================================"
    echo "  Step 0: Generate RLBench demonstrations"
    echo "  Train episodes: $TRAIN_EPISODES per task"
    echo "  Test episodes:  $TEST_EPISODES per task"
    echo "========================================"

    if [[ ! -d "$MANIGAUSSIAN_DIR/third_party/RLBench" ]]; then
        echo "ERROR: RLBench not found at $MANIGAUSSIAN_DIR/third_party/RLBench"
        echo "  Set MANIGAUSSIAN_DIR to your ManiGaussian checkout,"
        echo "  or set SKIP_GEN_DEMOS=1 if demos already exist."
        exit 1
    fi

    pushd "$MANIGAUSSIAN_DIR" > /dev/null

    for task in "${TASKS[@]}"; do
        echo ""
        echo "### Generating demonstrations for: $task"

        # Training data
        if [[ " ${SPLIT_ARR[*]} " == *" train_data "* ]]; then
            TRAIN_DIR="$RLBENCH_ROOT/train_data"
            EPISODE_DIR="$TRAIN_DIR/$task/all_variations/episodes"
            EXISTING=$(find "$EPISODE_DIR" -maxdepth 1 -name 'episode*' 2>/dev/null | wc -l)
            if [[ "$EXISTING" -ge "$TRAIN_EPISODES" ]]; then
                echo "  [SKIP] train_data/$task: $EXISTING episodes already exist"
            else
                echo "  Generating $TRAIN_EPISODES train episodes..."
                python third_party/RLBench/tools/dataset_generator.py \
                    --tasks="$task" \
                    --save_path="$TRAIN_DIR" \
                    --episodes_per_task="$TRAIN_EPISODES" \
                    --all_variations=True \
                    --image_size="$IMAGE_SIZE"
            fi
        fi

        # Test data
        if [[ " ${SPLIT_ARR[*]} " == *" test_data "* ]]; then
            TEST_DIR="$RLBENCH_ROOT/test_data"
            EPISODE_DIR="$TEST_DIR/$task/all_variations/episodes"
            EXISTING=$(find "$EPISODE_DIR" -maxdepth 1 -name 'episode*' 2>/dev/null | wc -l)
            if [[ "$EXISTING" -ge "$TEST_EPISODES" ]]; then
                echo "  [SKIP] test_data/$task: $EXISTING episodes already exist"
            else
                echo "  Generating $TEST_EPISODES test episodes..."
                python third_party/RLBench/tools/dataset_generator.py \
                    --tasks="$task" \
                    --save_path="$TEST_DIR" \
                    --episodes_per_task="$TEST_EPISODES" \
                    --all_variations=True \
                    --image_size="$IMAGE_SIZE"
            fi
        fi

        echo "  [DONE] $task"
    done

    popd > /dev/null
    echo ""
    echo "[Step 0 DONE] Demonstrations generated"
    echo ""
fi

cd "$LPWM_DIR"

# ==========================================================
#  STEP 1: Fuse multi-camera RGB+depth into PLY point clouds
# ==========================================================
echo "========================================"
echo "  Step 1: Generate PLY point clouds"
echo "========================================"

PYTHONPATH=. python -u scripts/rlbench_ply.py \
    --root "$RLBENCH_ROOT" \
    --splits ${SPLIT_ARR[*]} \
    --tasks ${TASKS[*]} \
    --cameras $CAMERAS \
    --depth-scale "$DEPTH_SCALE" \
    --max-points "$MAX_POINTS" \
    --fov "$FOV"

echo ""
echo "[Step 1 DONE] PLY point clouds generated"
echo ""

# ==========================================================
#  STEP 2: Voxelize PLY files into RGBO (4-channel) voxels
# ==========================================================
echo "========================================"
echo "  Step 2: Voxelize into RGBO"
echo "========================================"

PYTHONPATH=. python -u scripts/preprocess_rlbench_voxels.py \
    --root "$RLBENCH_ROOT" \
    --splits ${SPLIT_ARR[*]} \
    --tasks ${TASKS[*]} \
    --grid_whd $GRID_SIZE \
    --voxel_mode avg_rgb \
    --rgbo

echo ""
echo "[Step 2 DONE] RGBO voxels generated"

echo ""
echo "========================================"
echo "  All steps complete!"
echo "========================================"
echo ""
echo "  RGBO voxels are at:"
echo "    $RLBENCH_ROOT/{split}/{task}/all_variations/episodes/episodeN/voxel_cache/"
echo ""
echo "  Each voxel file is [4, D, H, W] (R, G, B, Occupancy)."
echo ""
echo "  To re-run with --force (rebuild existing caches), add --force"
echo "  to the preprocess_rlbench_voxels.py call above."
echo ""
