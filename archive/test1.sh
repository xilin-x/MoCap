#!/usr/bin/env bash
set -euo pipefail

blender -b --python fbx_to_video.py -- \
    --input "${MOCAP_DATA:-/path/to/mocap_data}/results/meshes/Cossack Squats/Cossack Squats (A)/test.fbx" \
    --output "${MOCAP_DATA:-/path/to/mocap_data}/results/meshes/Cossack Squats/Cossack Squats (A)/test.mp4" \
    --fps 30 \
    --view front \
    --keep-frames


# blender -b --python fbx_to_video.py -- \
#     --input "${MOCAP_DATA:-/path/to/mocap_data}/results/meshes/Arm Circles/Arm Circles (A)/test.fbx" \
#     --output "${MOCAP_DATA:-/path/to/mocap_data}/results/meshes/Arm Circles/Arm Circles (A)/test10.mp4" \
#     --fps 30 \
#     --view front \
#     --start 2 \
#     --end 11 \
#     --keep-frames