import os
import glob
import time

import cv2
import numpy as np

from notebook.utils import setup_sam_3d_body

if __name__ == "__main__":
    estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

    seqs_name = sorted(os.listdir("results/frames"))
    for seq_name in seqs_name:
        videos_name = sorted(os.listdir(os.path.join("results/frames", seq_name)))
        for video_name in videos_name:
            results_dir = os.path.join("results/bboxes", seq_name, video_name)
            os.makedirs(results_dir, exist_ok=True)
            frames = sorted(glob.glob(os.path.join("results/frames", seq_name, video_name, "*.jpg")))
            for frame in frames[-1:-3:-1]:  # Process last 3 frames in reverse order
                start_time = time.time()
                img_cv2 = cv2.imread(frame)
                img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
                outputs = estimator.process_one_image(frame)
                assert len(outputs) == 1, f"Expected exactly one person detected in {frame}, but got {len(outputs)}"
                # outputs dict save into numpy file
                output_path = os.path.join(results_dir, os.path.basename(frame).replace(".jpg", ".npy"))
                np.save(output_path, outputs[0])
                end_time = time.time()
                print(f"Processed {frame} in {end_time - start_time:.2f} seconds")
