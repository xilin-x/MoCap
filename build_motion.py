import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pymomentum.geometry as pym_geometry
from rich.progress import track

ROOT = Path(__file__).resolve().parent


def load_skeleton(assets, lod):
    character = pym_geometry.Character.load_fbx(
        str(assets / f"lod{lod}.fbx"), str(assets / "compact_v6_1.model"), load_blendshapes=False
    )
    skeleton = character.skeleton
    return (
        np.asarray(skeleton.joint_names), np.asarray(skeleton.joint_parents, dtype=np.int32),
        np.asarray(skeleton.offsets, dtype=np.float32), np.asarray(skeleton.pre_rotations, dtype=np.float32)
    )


def render_motion(coords, parents, names, out_dir, out_video, fps):
    out_dir.mkdir(parents=True, exist_ok=True)

    low = coords.min(axis=(0, 1))
    high = coords.max(axis=(0, 1))
    center = (low + high) / 2
    radius = (high - low).max() * 0.55

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=15, azim=-70)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    points = ax.scatter([], [], [], s=10)
    bones = [(p, i) for i, p in enumerate(parents) if p >= 0]
    lines = [ax.plot([], [], [])[0] for _ in bones]

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for t in track(range(len(coords)), description="Rendering skeleton"):
        xyz = coords[t]
        points._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])

        for line, (parent, child) in zip(lines, bones):
            line.set_data([xyz[parent, 0], xyz[child, 0]], [xyz[parent, 1], xyz[child, 1]])
            line.set_3d_properties([xyz[parent, 2], xyz[child, 2]])

        ax.set_title(names[t])
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(str(out_dir / f"{names[t]}.jpg"), image)
        writer.write(image)

    writer.release()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-a", "--assets", type=Path, default=ROOT / "lib" / "mhr" / "assets")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--lod", type=int, default=1)
    args = parser.parse_args()

    files = sorted(args.input.glob("frame_*.npz"))
    if not files:
        raise FileNotFoundError(f"No frame_*.npz found in {args.input}")

    coords, rots, names = [], [], []
    for file in track(files, description="Building motion"):
        data = np.load(file)
        coords.append(data["joint_coords"])
        rots.append(data["joint_rots"])
        names.append(file.stem)

    coords = np.stack(coords)
    rots = np.stack(rots)
    names = np.asarray(names)

    joint_names, parents, offsets, pre_rotations = load_skeleton(args.assets, args.lod)
    assert coords.shape[1] == len(joint_names)

    out_dir = args.input.parent
    np.savez(out_dir / "motion.npz", joint_coords=coords, joint_rots=rots, frame_names=names, fps=args.fps)
    np.savez(out_dir / "skeleton.npz", joint_names=joint_names, parents=parents, offsets=offsets, pre_rotations=pre_rotations)

    render_motion(coords, parents, names, out_dir / "skeleton_visualizations", out_dir / "skeleton_motion.mp4", args.fps)

    print(f"Saved: {out_dir / 'motion.npz'}")
    print(f"Saved: {out_dir / 'skeleton.npz'}")
    print(f"Saved: {out_dir / 'skeleton_motion.mp4'}")
