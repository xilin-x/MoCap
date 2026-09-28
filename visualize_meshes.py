import os
import glob
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import trimesh

from src.script_utils import iter_videos, progress_bar

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-m", "--meshes", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-v", "--video", nargs="+", default=["all"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--face-step", type=int, default=4)
    parser.add_argument("--elev", type=float, default=10)
    parser.add_argument("--azim", type=float, default=-90)
    args = parser.parse_args()

    with progress_bar() as progress:
        for seq_name, video_name, frame_dir in iter_videos(args.input, args.video):

            mesh_dir = os.path.join(args.meshes, seq_name, video_name, "meshes")

            output_dir = os.path.join(args.output, seq_name, video_name)

            os.makedirs(output_dir, exist_ok=True)

            frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))

            task = progress.add_task(f"{seq_name}/{video_name}", total=len(frames))

            for frame in frames:
                    name = os.path.splitext(os.path.basename(frame))[0]

                    mesh_path = os.path.join(mesh_dir, f"{name}_mesh_000.ply")

                    output_path = os.path.join(output_dir, f"{name}.jpg")

                    if args.resume and os.path.exists(output_path):
                        progress.advance(task)
                        continue

                    if not os.path.exists(mesh_path):
                        progress.console.print(f"[red]ERROR: Missing mesh: {mesh_path}")
                        progress.advance(task)
                        continue

                    try:
                        mesh = trimesh.load(mesh_path, process=False)

                        vertices = mesh.vertices
                        faces = mesh.faces[::args.face_step]

                        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2

                        radius = (vertices.max(axis=0) - vertices.min(axis=0)).max() / 2 * 1.1

                        fig = plt.figure(figsize=(6, 6))
                        ax = fig.add_subplot(111, projection="3d")

                        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, linewidth=0)

                        ax.set_xlim(center[0] - radius, center[0] + radius)
                        ax.set_ylim(center[1] - radius, center[1] + radius)
                        ax.set_zlim(center[2] - radius, center[2] + radius)

                        ax.set_box_aspect([1, 1, 1])
                        ax.view_init(elev=args.elev, azim=args.azim)
                        ax.set_axis_off()

                        plt.savefig(output_path, dpi=120, bbox_inches="tight", pad_inches=0)

                        plt.close(fig)

                    except Exception as e:
                        progress.console.print(f"[red]ERROR: {mesh_path}: {e}")

                    progress.advance(task)
