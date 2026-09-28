#!/usr/bin/env bash
set -euo pipefail

blender -b --python export_fbx_blender.py -- \
    --template "/hdd/xilin/Datasets/mocap_data/templates/TPose.fbx" \
    --motion "/hdd/xilin/Datasets/mocap_data/results/meshes/Cossack Squats/Cossack Squats (A)/motion.npz" \
    --skeleton "/hdd/xilin/Datasets/mocap_data/results/meshes/Cossack Squats/Cossack Squats (A)/skeleton.npz" \
    --output "/hdd/xilin/Datasets/mocap_data/results/meshes/Cossack Squats/Cossack Squats (A)/test.fbx"

echo "=== Done ==="
