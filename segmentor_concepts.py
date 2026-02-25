import os
import glob
import time

import torch
import numpy as np
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

if __name__ == "__main__":
    sam3_root = os.path.split(os.path.dirname(sam3.__file__))[0]
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    seqs_name = sorted(os.listdir("results/frames"))
    for seq_name in seqs_name[:1]:
        videos_name = sorted(os.listdir(os.path.join("results/frames", seq_name)))
        for video_name in videos_name[2:]:
            masks_dir = os.path.join("results/masks", seq_name, video_name)
            os.makedirs(masks_dir, exist_ok=True)
            frames = sorted(glob.glob(os.path.join("results/frames", seq_name, video_name, "*.jpg")))
            for frame in frames[515:]:
                torch.cuda.empty_cache()
                start_time = time.time()
                image = Image.open(frame)
                inference_state = processor.set_image(image)
                processor.reset_all_prompts(inference_state)
                inference_state = processor.set_text_prompt(state=inference_state, prompt="person")
                image_display = Image.open(frame)
                plot_results(image_display, inference_state)
                mask = inference_state['masks'].cpu().numpy().squeeze()
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                mask_image.save(os.path.join(masks_dir, os.path.basename(frame).replace(".jpg", ".png")))
                end_time = time.time()
                print(f"Processed {frame} in {end_time - start_time:.2f} seconds")
