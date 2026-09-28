import os
import glob
import argparse
import cv2
import numpy as np
import trimesh

os.environ["MOMENTUM_ENABLED"] = "0"

from src.script_utils import iter_videos, progress_bar, setup_paths, setup_torch

ROOT = os.path.dirname(os.path.abspath(__file__))
setup_paths(ROOT)
setup_torch()

from body_model.notebook.utils import setup_sam_3d_body, process_image_with_mask, setup_visualizer, visualize_2d_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-m", "--masks", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-v", "--video", nargs="+", default=["all"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")
    visualizer = setup_visualizer()

    with progress_bar() as progress:
        for seq_name, video_name, frame_dir in iter_videos(args.input, args.video):
            mask_dir = os.path.join(args.masks, seq_name, video_name)
            video_dir = os.path.join(args.output, seq_name, video_name)

            param_dir = os.path.join(video_dir, "mhr_params")
            mesh_dir = os.path.join(video_dir, "meshes")
            vis_dir = os.path.join(video_dir, "visualizations")

            os.makedirs(param_dir, exist_ok=True)
            os.makedirs(mesh_dir, exist_ok=True)
            os.makedirs(vis_dir, exist_ok=True)

            frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))

            task = progress.add_task(f"{seq_name}/{video_name}", total=len(frames))

            for frame in frames:
                    name = os.path.splitext(os.path.basename(frame))[0]

                    mask_path = os.path.join(mask_dir, f"{name}.png")
                    param_path = os.path.join(param_dir, f"{name}.npz")
                    mesh_path = os.path.join(mesh_dir, f"{name}_mesh_000.ply")
                    vis_path = os.path.join(vis_dir, f"{name}.jpg")

                    if (args.resume and os.path.exists(param_path) and os.path.exists(mesh_path) and os.path.exists(vis_path)):
                        progress.advance(task)
                        continue

                    if not os.path.exists(mask_path):
                        progress.console.print(f"[red]ERROR: Missing mask: {mask_path}")
                        progress.advance(task)
                        continue

                    try:
                        image = cv2.imread(frame)

                        output = process_image_with_mask(estimator, frame, mask_path)

                        if not output:
                            progress.console.print(f"[yellow]WARNING: No result: {frame}")
                            progress.advance(task)
                            continue

                        data = output[0]

                        np.savez(
                            param_path,
                            mhr_model_params=data["mhr_model_params"],
                            joint_coords=data["pred_joint_coords"],
                            joint_rots=data["pred_global_rots"],
                            global_rot=data["global_rot"],
                            body_pose=data["body_pose_params"],
                            hand_pose=data["hand_pose_params"],
                            shape=data["shape_params"],
                            scale=data["scale_params"],
                            cam_t=data["pred_cam_t"]
                        )

                        vertices = (data["pred_vertices"].copy() + data["pred_cam_t"])

                        mesh = trimesh.Trimesh(vertices=vertices, faces=estimator.faces, process=False)

                        rot = trimesh.transformations.rotation_matrix(
                            np.radians(180),
                            [1, 0, 0],
                        )
                        mesh.apply_transform(rot)
                        mesh.export(mesh_path)

                        vis = visualize_2d_results(image, output, visualizer)

                        if vis:
                            cv2.imwrite(vis_path, vis[0])

                    except Exception as e:
                        progress.console.print(f"[red]ERROR: {frame}: {e}")

                    progress.advance(task)
