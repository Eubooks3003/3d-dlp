#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/home/ellina/Desktop/lpwm-copy/scripts/hdf5_states_to_voxels.py"
CORE_DIR="/home/ellina/Desktop/data/3D-DLP-mimicgen-data/core"
DATA_DIR="/home/ellina/Desktop/data/3D-DLP-mimicgen-data"
MAX_JOBS=4

run_task() {
    local hdf5="$1"
    local task=$(basename "$hdf5" .hdf5)
    local output_dir="$DATA_DIR/${task}/voxel_cache"
    local log_file="$DATA_DIR/${task}_voxel.log"

    echo "[START] $task"
    python "$SCRIPT" \
        --input "$hdf5" \
        --output-dir "$output_dir" \
        --use-task-bounds \
        --compress \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        echo "[DONE]  $task"
    else
        echo "[FAIL]  $task — see $log_file"
    fi
}

export -f run_task
export SCRIPT CORE_DIR DATA_DIR

ls "$CORE_DIR"/*.hdf5 | xargs -P "$MAX_JOBS" -I {} bash -c 'run_task "$@"' _ {}

echo "All tasks complete. Logs: $DATA_DIR/*_voxel.log"
