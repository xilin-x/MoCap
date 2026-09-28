import os
import glob
import argparse

import torch
import numpy as np
from PIL import Image
from src.script_utils import iter_videos, progress_bar, setup_paths, setup_torch

ROOT = os.path.dirname(os.path.abspath(__file__))
setup_paths(ROOT)
setup_torch()

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-v", "--video", nargs="+", default=["all"])
    args = parser.parse_args()

    bpe_path = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")

    model = build_sam3_image_model(bpe_path=bpe_path)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=0.5)

    with progress_bar() as progress:
        for seq_name, video_name, frame_dir in iter_videos(args.input, args.video):
            masks_dir = os.path.join(args.output, seq_name, video_name)
            os.makedirs(masks_dir, exist_ok=True)
            frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
            task = progress.add_task(f"{seq_name}/{video_name}", total=len(frames))

            for frame in frames:
                    image = Image.open(frame).convert("RGB")

                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        state = processor.set_image(image)
                        processor.reset_all_prompts(state)
                        state = processor.set_text_prompt(state=state, prompt="person")

                    mask = state["masks"].cpu().numpy().squeeze()

                    Image.fromarray((mask * 255).astype(np.uint8)
                                   ).save(os.path.join(masks_dir,
                                                       os.path.basename(frame).replace(".jpg", ".png")))

                    progress.update(task, advance=1)
