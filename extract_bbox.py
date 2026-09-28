import os
import glob
import argparse
import numpy as np
from PIL import Image

import torch

os.environ["MOMENTUM_ENABLED"] = "0"

ROOT = os.path.dirname(os.path.abspath(__file__))
from src.script_utils import iter_videos, progress_bar, setup_paths, setup_torch

setup_paths(ROOT)
setup_torch()

from body_model.notebook.utils import setup_sam_3d_body

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-v", "--video", nargs="+", default=["all"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

    bpe_path = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
    sam3_model = build_sam3_image_model(bpe_path=bpe_path)
    sam3_model.eval()
    processor = Sam3Processor(sam3_model, confidence_threshold=0.5)

    with progress_bar() as progress:
        for seq_name, video_name, frame_dir in iter_videos(args.input, args.video):
            output_dir = os.path.join(args.output, seq_name, video_name)
            os.makedirs(output_dir, exist_ok=True)
            frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
            task = progress.add_task(f"{seq_name}/{video_name}", total=len(frames))

            for frame in frames:
                    name = os.path.splitext(os.path.basename(frame))[0]
                    output_path = os.path.join(output_dir, f"{name}.npy")

                    if args.resume and os.path.exists(output_path):
                        progress.advance(task)
                        continue

                    saved = False

                    try:
                        outputs = estimator.process_one_image(frame)

                        if len(outputs) > 0:
                            if len(outputs) > 1:
                                progress.console.print(f"[yellow]WARNING: {len(outputs)} people, using first: {frame}")

                            np.save(output_path, outputs[0])
                            saved = True
                        else:
                            progress.console.print(f"[yellow]WARNING: Detection failed, trying text mask: {frame}")

                    except Exception as e:
                        progress.console.print(f"[yellow]WARNING: Detection error, trying text mask: {frame} | {e}")

                    if not saved:
                        try:
                            image = Image.open(frame).convert("RGB")

                            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                                state = processor.set_image(image)
                                processor.reset_all_prompts(state)
                                state = processor.set_text_prompt(state=state, prompt="person")

                            mask = state["masks"].cpu().numpy().squeeze()
                            bbox = mask_to_bbox(mask)

                            if bbox is not None:
                                np.save(output_path, {"bbox": bbox, "source": "text_mask"})
                                saved = True
                            else:
                                progress.console.print(f"[red]ERROR: Text mask failed: {frame}")

                        except Exception as e:
                            progress.console.print(f"[red]ERROR: Text mask error: {frame} | {e}")

                    progress.advance(task)
