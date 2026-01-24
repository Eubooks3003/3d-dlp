#!/usr/bin/env bash
#
# batch_render.sh
#
# Batch render multiple examples and assemble turntable videos.
#
# Usage:
#   ./batch_render.sh /path/to/paper_meshes /path/to/renders [OPTIONS]
#
# Options:
#   --preset PRESET     paper|web|vertex_color (default: paper)
#   --frames N          turntable frames (default: 120)
#   --fps N             video framerate (default: 30)
#   --res W H           resolution (default: 1920 1080)
#   --engine ENGINE     CYCLES|EEVEE (default: CYCLES)
#   --device DEVICE     GPU|CPU (default: GPU)
#   --blender PATH      path to blender executable (default: blender)
#   --parallel N        number of parallel renders (default: 1)
#   --skip-video        skip video assembly
#   --only-video        only assemble videos (skip rendering)
#

set -e

# Default values
PRESET="paper"
FRAMES=120
FPS=30
RES_W=1920
RES_H=1080
ENGINE="CYCLES"
DEVICE="GPU"
BLENDER="blender"
PARALLEL=1
SKIP_VIDEO=false
ONLY_VIDEO=false

# Parse arguments
INPUT_DIR=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --frames)
            FRAMES="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --res)
            RES_W="$2"
            RES_H="$3"
            shift 3
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --blender)
            BLENDER="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --skip-video)
            SKIP_VIDEO=true
            shift
            ;;
        --only-video)
            ONLY_VIDEO=true
            shift
            ;;
        -h|--help)
            head -30 "$0" | tail -25
            exit 0
            ;;
        *)
            if [[ -z "$INPUT_DIR" ]]; then
                INPUT_DIR="$1"
            elif [[ -z "$OUTPUT_DIR" ]]; then
                OUTPUT_DIR="$1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$INPUT_DIR" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 INPUT_DIR OUTPUT_DIR [OPTIONS]"
    echo "Run '$0 --help' for more info."
    exit 1
fi

# Get script directory (for finding render_blender.py)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER_SCRIPT="$SCRIPT_DIR/render_blender.py"

if [[ ! -f "$RENDER_SCRIPT" ]]; then
    echo "[ERROR] Cannot find render_blender.py at $RENDER_SCRIPT"
    exit 1
fi

# Check for blender
if ! command -v "$BLENDER" &> /dev/null; then
    echo "[ERROR] Blender not found. Install Blender or specify path with --blender"
    exit 1
fi

# Check for ffmpeg (for video assembly)
if ! command -v ffmpeg &> /dev/null; then
    echo "[WARNING] ffmpeg not found. Video assembly will be skipped."
    SKIP_VIDEO=true
fi

echo "=============================================="
echo "Batch Render Configuration"
echo "=============================================="
echo "Input:     $INPUT_DIR"
echo "Output:    $OUTPUT_DIR"
echo "Preset:    $PRESET"
echo "Engine:    $ENGINE"
echo "Device:    $DEVICE"
echo "Res:       ${RES_W}x${RES_H}"
echo "Frames:    $FRAMES"
echo "FPS:       $FPS"
echo "Parallel:  $PARALLEL"
echo "=============================================="

mkdir -p "$OUTPUT_DIR"

# Find all example directories (those containing .ply files)
EXAMPLES=()
for dir in "$INPUT_DIR"/*/; do
    if [[ -f "$dir/gt.ply" ]] || [[ -f "$dir/rec.ply" ]]; then
        EXAMPLES+=("$dir")
    fi
done

# Also check if input_dir itself contains PLY files
if [[ -f "$INPUT_DIR/gt.ply" ]] || [[ -f "$INPUT_DIR/rec.ply" ]]; then
    EXAMPLES+=("$INPUT_DIR")
fi

echo "Found ${#EXAMPLES[@]} examples to render"

# Function to render a single example
render_example() {
    local input_path="$1"
    local example_name=$(basename "$input_path")
    local output_path="$OUTPUT_DIR/$example_name"

    echo "[render] Processing: $example_name"

    "$BLENDER" -b -P "$RENDER_SCRIPT" -- \
        --input_dir "$input_path" \
        --output_dir "$output_path" \
        --preset "$PRESET" \
        --engine "$ENGINE" \
        --device "$DEVICE" \
        --res "$RES_W" "$RES_H" \
        --frames "$FRAMES"

    echo "[render] Done: $example_name"
}

# Function to assemble turntable video
assemble_video() {
    local frames_dir="$1"
    local output_video="$2"

    if [[ ! -d "$frames_dir" ]]; then
        echo "[video] No frames directory: $frames_dir"
        return
    fi

    local frame_count=$(ls -1 "$frames_dir"/frame_*.png 2>/dev/null | wc -l)
    if [[ $frame_count -eq 0 ]]; then
        echo "[video] No frames in: $frames_dir"
        return
    fi

    echo "[video] Assembling $frame_count frames -> $output_video"

    ffmpeg -y -framerate "$FPS" \
        -i "$frames_dir/frame_%04d.png" \
        -c:v libx264 \
        -pix_fmt yuv420p \
        -crf 18 \
        "$output_video" \
        2>/dev/null

    echo "[video] Created: $output_video"
}

# Function to create side-by-side comparison video
create_comparison_video() {
    local gt_video="$1"
    local rec_video="$2"
    local output_video="$3"

    if [[ ! -f "$gt_video" ]] || [[ ! -f "$rec_video" ]]; then
        echo "[video] Missing videos for comparison"
        return
    fi

    echo "[video] Creating comparison: $output_video"

    ffmpeg -y \
        -i "$gt_video" \
        -i "$rec_video" \
        -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
        -map "[v]" \
        -c:v libx264 \
        -pix_fmt yuv420p \
        -crf 18 \
        "$output_video" \
        2>/dev/null

    echo "[video] Created: $output_video"
}

# Render phase
if [[ "$ONLY_VIDEO" == false ]]; then
    if [[ $PARALLEL -gt 1 ]]; then
        # Parallel rendering
        echo "[render] Running $PARALLEL parallel renders..."
        export -f render_example
        export BLENDER RENDER_SCRIPT PRESET ENGINE DEVICE RES_W RES_H FRAMES OUTPUT_DIR
        printf '%s\n' "${EXAMPLES[@]}" | xargs -P "$PARALLEL" -I {} bash -c 'render_example "$@"' _ {}
    else
        # Sequential rendering
        for example in "${EXAMPLES[@]}"; do
            render_example "$example"
        done
    fi
fi

# Video assembly phase
if [[ "$SKIP_VIDEO" == false ]]; then
    echo ""
    echo "=============================================="
    echo "Assembling Videos"
    echo "=============================================="

    for example in "${EXAMPLES[@]}"; do
        example_name=$(basename "$example")
        output_path="$OUTPUT_DIR/$example_name"

        # Assemble GT turntable
        if [[ -d "$output_path/turntable_gt" ]]; then
            assemble_video "$output_path/turntable_gt" "$output_path/turntable_gt.mp4"
        fi

        # Assemble REC turntable
        if [[ -d "$output_path/turntable_rec" ]]; then
            assemble_video "$output_path/turntable_rec" "$output_path/turntable_rec.mp4"
        fi

        # Create comparison video if both exist
        if [[ -f "$output_path/turntable_gt.mp4" ]] && [[ -f "$output_path/turntable_rec.mp4" ]]; then
            create_comparison_video \
                "$output_path/turntable_gt.mp4" \
                "$output_path/turntable_rec.mp4" \
                "$output_path/compare_gt_rec.mp4"
        fi
    done
fi

echo ""
echo "=============================================="
echo "Batch Render Complete!"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Print summary
echo "Contents:"
find "$OUTPUT_DIR" -name "*.png" | wc -l | xargs -I {} echo "  PNG images: {}"
find "$OUTPUT_DIR" -name "*.mp4" | wc -l | xargs -I {} echo "  MP4 videos: {}"
