import sys
import argparse
import shutil
import subprocess
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args():
    args = sys.argv[sys.argv.index("--") + 1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--view", choices=["front", "back", "left", "right"], default="front")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--point-size", type=int, default=1)
    parser.add_argument("--keep-frames", action="store_true")

    return parser.parse_args(args)


def get_animation_range(scene):
    frames = []

    for obj in scene.objects:
        if obj.animation_data is None:
            continue

        action = obj.animation_data.action

        if action is None:
            continue

        for curve in action.fcurves:
            for key in curve.keyframe_points:
                frames.append(key.co.x)

    if not frames:
        return scene.frame_start, scene.frame_end

    return int(min(frames)), int(max(frames))


def get_mesh_object():
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    if not meshes:
        raise RuntimeError("No mesh found")

    return max(meshes, key=lambda x: len(x.data.vertices))


def get_armature():
    arms = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]

    if not arms:
        raise RuntimeError("No armature found")

    return arms[0]


def evaluated_vertices(obj, depsgraph):
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    try:
        vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float32)

        mesh.vertices.foreach_get("co", vertices)

        vertices = vertices.reshape(-1, 3)

        matrix = np.array(eval_obj.matrix_world, dtype=np.float32)

        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]

        vertices = (vertices @ rotation.T + translation)

        return vertices

    finally:
        eval_obj.to_mesh_clear()


def bone_segments(arm):
    matrix = arm.matrix_world

    segments = []

    for bone in arm.pose.bones:
        head = matrix @ bone.head
        tail = matrix @ bone.tail

        segments.append((np.array(head, dtype=np.float32), np.array(tail, dtype=np.float32)))

    return segments


def project(points, view):
    if view == "front":
        x = points[:, 0]
        y = points[:, 2]

    elif view == "back":
        x = -points[:, 0]
        y = points[:, 2]

    elif view == "left":
        x = points[:, 1]
        y = points[:, 2]

    else:
        x = -points[:, 1]
        y = points[:, 2]

    return np.stack([x, y], axis=1)


def fit_bounds(points, margin=0.12):
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)

    center = (min_xy + max_xy) / 2.0

    size = max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])

    size *= 1.0 + margin * 2.0

    return center, size


def to_pixels(points, center, size, width, height):
    scale = min(width, height) / size

    x = ((points[:, 0] - center[0]) * scale + width / 2)

    y = (height / 2 - (points[:, 1] - center[1]) * scale)

    return np.stack([x, y], axis=1).astype(int)


def draw_points(image, points, point_size, value):
    h, w = image.shape

    for dx in range(-point_size, point_size + 1):
        for dy in range(-point_size, point_size + 1):
            x = points[:, 0] + dx
            y = points[:, 1] + dy

            valid = ((x >= 0) & (x < w) & (y >= 0) & (y < h))

            image[y[valid], x[valid]] = value


def draw_line(image, p0, p1, value=0, thickness=2):
    x0, y0 = map(int, p0)
    x1, y1 = map(int, p1)

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    steps = max(dx, dy, 1)

    xs = np.linspace(x0, x1, steps + 1).astype(int)

    ys = np.linspace(y0, y1, steps + 1).astype(int)

    h, w = image.shape

    for offset_x in range(-thickness, thickness + 1):
        for offset_y in range(-thickness, thickness + 1):
            x = xs + offset_x
            y = ys + offset_y

            valid = ((x >= 0) & (x < w) & (y >= 0) & (y < h))

            image[y[valid], x[valid]] = value


def save_ppm(path, image):
    h, w = image.shape

    rgb = np.repeat(image[:, :, None], 3, axis=2)

    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(rgb.tobytes())


def make_video(frame_dir, output, fps):
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RuntimeError(f"ffmpeg not found. Frames saved at {frame_dir}")

    cmd = [
        ffmpeg, "-y", "-framerate",
        str(fps), "-i",
        str(frame_dir / "frame_%04d.ppm"), "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        str(output)
    ]

    print("\nCreating MP4...", flush=True)

    subprocess.run(cmd, check=True)

    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    frame_dir = (args.output.parent / f"{args.output.stem}_frames")

    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading FBX: {args.input}", flush=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    bpy.ops.import_scene.fbx(filepath=str(args.input))

    scene = bpy.context.scene

    start, end = get_animation_range(scene)

    if args.start is not None:
        start = args.start

    if args.end is not None:
        end = args.end

    scene.frame_start = start
    scene.frame_end = end

    mesh_obj = get_mesh_object()
    arm = get_armature()

    depsgraph = bpy.context.evaluated_depsgraph_get()

    scene.frame_set(start)
    bpy.context.view_layer.update()

    first_vertices = evaluated_vertices(mesh_obj, depsgraph)

    first_projected = project(first_vertices, args.view)

    center, size = fit_bounds(first_projected, margin=0.15)

    print(f"Mesh: {mesh_obj.name}", flush=True)

    print(f"Armature: {arm.name}", flush=True)

    print(f"Vertices: {len(first_vertices)}", flush=True)

    print(f"Frames: {start} - {end}", flush=True)

    print(f"Total: {end - start + 1}", flush=True)

    print(f"FPS: {args.fps}", flush=True)

    print("\nStart FBX evaluation...\n", flush=True)

    total = end - start + 1

    for i, frame in enumerate(range(start, end + 1)):
        print(f"[{i + 1}/{total}] "
              f"Frame {frame}", flush=True)

        scene.frame_set(frame)
        bpy.context.view_layer.update()

        vertices = evaluated_vertices(mesh_obj, depsgraph)

        projected = project(vertices, args.view)

        pixels = to_pixels(projected, center, size, args.width, args.height)

        image = np.full((
            args.height,
            args.width,
        ), 255, dtype=np.uint8)

        draw_points(image, pixels, args.point_size, 170)

        segments = bone_segments(arm)

        for head, tail in segments:
            pair = np.stack([head, tail], axis=0)

            pair = project(pair, args.view)

            pair = to_pixels(pair, center, size, args.width, args.height)

            draw_line(image, pair[0], pair[1], value=0, thickness=1)

        path = (frame_dir / f"frame_{i:04d}.ppm")

        save_ppm(path, image)

    print("\nAll FBX frames evaluated.", flush=True)

    make_video(frame_dir, args.output, args.fps)

    if not args.keep_frames:
        shutil.rmtree(frame_dir)

    print("\nDone.", flush=True)
