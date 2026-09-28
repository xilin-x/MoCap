#!/usr/bin/env bash
set -euo pipefail

blender -b --python export_fbx_blender.py -- \
    --template "${MOCAP_DATA:-/path/to/mocap_data}/templates/TPose.fbx" \
    --motion "${MOCAP_DATA:-/path/to/mocap_data}/results/meshes/Cossack Squats/Cossack Squats (A)/motion.npz" \
    --skeleton "${MOCAP_DATA:-/path/to/mocap_data}/results/meshes/Cossack Squats/Cossack Squats (A)/skeleton.npz" \
    --output "${MOCAP_DATA:-/path/to/mocap_data}/results/meshes/Cossack Squats/Cossack Squats (A)/test.fbx"

echo "=== Done ==="
