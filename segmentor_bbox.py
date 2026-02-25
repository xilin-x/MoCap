import os
import glob
import time

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sam3
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

if __name__ == "__main__":
    sam3_root = os.path.split(os.path.dirname(sam3.__file__))[0]
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    seqs_name = sorted(os.listdir("results/frames"))
    for seq_name in seqs_name:
        videos_name = sorted(os.listdir(os.path.join("results/frames", seq_name)))
        for video_name in videos_name:
            masks_dir = os.path.join("results/masks", seq_name, video_name)
            os.makedirs(masks_dir, exist_ok=True)
            frames = sorted(glob.glob(os.path.join("results/frames", seq_name, video_name, "*.jpg")))
            bboxes = sorted(glob.glob(os.path.join("results/bboxes", seq_name, video_name, "*.npy")))
            for bbox, frame in zip(bboxes[-1:-3:-1], frames[-1:-3:-1]):
                start_time = time.time()
                bbox_data = np.ceil(np.load(bbox, allow_pickle=True).item()['bbox'])
                image = Image.open(frame)
                width, height = image.size
                box_input_xywh = torch.tensor([bbox_data[0], bbox_data[1], bbox_data[2] - bbox_data[0], bbox_data[3] - bbox_data[1]]).view(-1, 4)
                inference_state = processor.set_image(image)
                box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
                norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
                processor.reset_all_prompts(inference_state)
                inference_state = processor.add_geometric_prompt(state=inference_state, box=norm_box_cxcywh, label=True)
                image_display = Image.open(frame)
                image_with_box = draw_box_on_image(image_display, box_input_xywh.flatten().tolist())
                # plt.imshow(image_with_box)
                # plt.axis("off")  # Hide the axis
                # plt.show()
                # plot_results(image_display, inference_state)
                mask = inference_state['masks'].cpu().numpy().squeeze()
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                mask_image.save(os.path.join(masks_dir, os.path.basename(frame).replace(".jpg", ".png")))
                end_time = time.time()
                print(f"Processed {frame} in {end_time - start_time:.2f} seconds")
