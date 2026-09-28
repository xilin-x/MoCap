uv run --no-sync python frames_to_video.py \
    --input "${MOCAP_DATA:-/path/to/mocap_data}/results/meshes/Cossack Squats/Cossack Squats (A)/test_frames" \
    --output "${MOCAP_DATA:-/path/to/mocap_data}/results/meshes/Cossack Squats/Cossack Squats (A)/test.mp4" \
    --fps 30