import os
import glob
import time

import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from notebook.utils import (
    setup_sam_3d_body, setup_visualizer, visualize_2d_results, visualize_3d_mesh, save_mesh_results, display_results_grid,
    process_image_with_mask
)


if __name__ == "__main__":
    estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")
    seqs_name = sorted(os.listdir("results/frames"))
    for seq_name in seqs_name[:1]:
        videos_name = sorted(os.listdir(os.path.join("results/frames", seq_name)))
        for video_name in videos_name:
            meshes_dir = os.path.join("results/meshes", seq_name, video_name)
            os.makedirs(meshes_dir, exist_ok=True)
            frames = sorted(glob.glob(os.path.join("results/frames", seq_name, video_name, "*.jpg")))
            masks = sorted(glob.glob(os.path.join("results/masks", seq_name, video_name, "*.png")))
            for mask, frame in zip(masks, frames):
                start_time = time.time()
                image_name = os.path.basename(frame).split(".")[0]
                image_display = cv2.imread(frame)
                output = process_image_with_mask(estimator, frame, mask)
                mesh = visualize_3d_mesh(image_display, output, estimator.faces)
                # for i, combined_img in enumerate(mesh):
                #     combined_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
                #     plt.figure(figsize=(20, 5))
                #     plt.imshow(combined_img)
                #     plt.title(f'Mask-Based Person {i}: Original | Mesh Overlay | Front View | Side View')
                #     plt.axis('off')
                #     plt.show()
                mask_ply_files = save_mesh_results(image_display, output, estimator.faces, meshes_dir, f"{image_name}")
                print(f"Saved mask-based results to: {meshes_dir}")
                end_time = time.time()
                print(f"Processed {frame} in {end_time - start_time:.2f} seconds")