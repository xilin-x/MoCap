#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATA="${MOCAP_DATA:?Set MOCAP_DATA to your dataset directory}"
GPU=0
FPS=30
TEMPLATE="$DATA/templates/TPose.fbx"

export CUDA_VISIBLE_DEVICES="$GPU"
export MOMENTUM_ENABLED=0

if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: Template not found: $TEMPLATE"
    exit 1
fi

echo "=== Extract frames ==="
uv run --no-sync python extract_frames.py \
    -i "$DATA/videos" \
    -o "$DATA/results/frames" \
    -w 8 \
    --resume

echo "=== Extract bboxes ==="
uv run --no-sync python extract_bbox.py \
    -i "$DATA/results/frames" \
    -o "$DATA/results/bboxes" \
    -v all \
    --resume

echo "=== Check bboxes ==="
uv run --no-sync python check_bboxes.py \
    -i "$DATA/results/frames" \
    -b "$DATA/results/bboxes"

echo "=== Extract masks ==="
uv run --no-sync python extract_masks.py \
    -i "$DATA/results/frames" \
    -b "$DATA/results/bboxes" \
    -o "$DATA/results/masks" \
    -v all \
    --resume

echo "=== Extract MHR ==="
uv run --no-sync python extract_meshes.py \
    -i "$DATA/results/frames" \
    -m "$DATA/results/masks" \
    -o "$DATA/results/meshes" \
    -v all \
    --resume

echo "=== Visualize meshes ==="
uv run --no-sync python visualize_meshes.py \
    -i "$DATA/results/frames" \
    -m "$DATA/results/meshes" \
    -o "$DATA/results/mesh_visualizations" \
    -v all \
    --resume

echo "=== Create mesh videos ==="
uv run --no-sync python images_to_video.py \
    -i "$DATA/results/mesh_visualizations" \
    -o "$DATA/results/mesh_videos" \
    --fps "$FPS" \
    --resume

echo "=== Build motion and FBX ==="

while IFS= read -r -d '' PARAMS; do
    OUT="$(dirname "$PARAMS")"
    NAME="$(basename "$OUT")"

    echo "Processing: $NAME"

    uv run --no-sync python build_motion.py \
        -i "$PARAMS" \
        --fps "$FPS"

    blender -b --python export_fbx_blender.py -- \
        --template "$TEMPLATE" \
        --motion "$OUT/motion.npz" \
        --skeleton "$OUT/skeleton.npz" \
        --output "$OUT/$NAME.fbx"

done < <(find "$DATA/results/meshes" -type d -name mhr_params -print0)

echo "=== Done ==="
