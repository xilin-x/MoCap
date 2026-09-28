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
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox

def get_mask(masks):
    masks = masks.detach().cpu().numpy()
    masks = np.squeeze(masks)

    if masks.ndim == 2:
        return masks

    if masks.ndim == 3:
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        return masks[np.argmax(areas)]

    raise ValueError(f"Invalid mask shape: {masks.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-b", "--bboxes", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-v", "--video", nargs="+", default=["all"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    bpe_path = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")

    model = build_sam3_image_model(bpe_path=bpe_path)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=0.5)

    with progress_bar() as progress:
        for seq_name, video_name, frame_dir in iter_videos(args.input, args.video):
            bbox_dir = os.path.join(args.bboxes, seq_name, video_name)
            mask_dir = os.path.join(args.output, seq_name, video_name)
            os.makedirs(mask_dir, exist_ok=True)
            frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
            task = progress.add_task(f"{seq_name}/{video_name}", total=len(frames))

            for frame in frames:
                    name = os.path.splitext(os.path.basename(frame))[0]

                    bbox_path = os.path.join(bbox_dir, f"{name}.npy")
                    mask_path = os.path.join(mask_dir, f"{name}.png")

                    if args.resume and os.path.exists(mask_path):
                        progress.advance(task)
                        continue

                    if not os.path.exists(bbox_path):
                        progress.console.print(f"[red]ERROR: Missing bbox: {frame}")
                        progress.advance(task)
                        continue

                    try:
                        bbox = np.ceil(np.load(bbox_path, allow_pickle=True).item()["bbox"])

                        image = Image.open(frame).convert("RGB")
                        width, height = image.size

                        box_xywh = torch.tensor([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]).view(-1, 4)

                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                            state = processor.set_image(image)

                            box = box_xywh_to_cxcywh(box_xywh)
                            box = normalize_bbox(box, width, height).flatten().tolist()

                            processor.reset_all_prompts(state)

                            state = processor.add_geometric_prompt(state=state, box=box, label=True)

                        mask = get_mask(state["masks"])

                        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

                    except Exception as e:
                        progress.console.print(f"[red]ERROR: {frame}: {e}")

                    progress.advance(task)
