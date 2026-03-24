#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/home/ellina/Desktop/lpwm-dev/scripts/hdf5_states_to_voxels.py"
CORE_DIR="/home/ellina/Desktop/data/3D-DLP-mimicgen-data/core"
DATA_DIR="/home/ellina/Desktop/data/3D-DLP-mimicgen-data"
MAX_JOBS=4
MAX_DEMOS=${MAX_DEMOS:-}  # e.g. MAX_DEMOS=200 ./run_all_voxels.sh

# If TASKS is set (space-separated), only process those. Otherwise process all hdf5s in CORE_DIR.
# Example: TASKS="mug_cleanup_d1 coffee_d2" ./run_all_voxels.sh
TASKS=${TASKS:-}

run_task() {
    local hdf5="$1"
    local task=$(basename "$hdf5" .hdf5)
    local output_dir="$DATA_DIR/${task}/voxel_cache"
    local log_file="$DATA_DIR/${task}_voxel.log"

    local extra_args=""
    if [[ -n "$MAX_DEMOS" ]]; then
        extra_args="--n $MAX_DEMOS"
    fi

    echo "[START] $task"
    python "$SCRIPT" \
        --input "$hdf5" \
        --output-dir "$output_dir" \
        --use-task-bounds \
        --compress \
        $extra_args \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        echo "[DONE]  $task"
    else
        echo "[FAIL]  $task — see $log_file"
    fi
}

export -f run_task
export SCRIPT CORE_DIR DATA_DIR MAX_DEMOS

if [[ -n "$TASKS" ]]; then
    for task in $TASKS; do
        echo "$CORE_DIR/${task}.hdf5"
    done | xargs -P "$MAX_JOBS" -I {} bash -c 'run_task "$@"' _ {}
else
    ls "$CORE_DIR"/*.hdf5 | xargs -P "$MAX_JOBS" -I {} bash -c 'run_task "$@"' _ {}
fi

echo "All tasks complete. Logs: $DATA_DIR/*_voxel.log"
