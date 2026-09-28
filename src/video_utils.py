from pathlib import Path

import cv2


def write_video(paths, output: Path, fps: float, progress=None, resize=False):
    first = cv2.imread(str(paths[0]))
    if first is None:
        raise RuntimeError(f"Cannot read: {paths[0]}")

    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {output}")

    try:
        for index, path in enumerate(paths):
            if progress:
                progress(index, path)
            image = cv2.imread(str(path))
            if image is None:
                print(f"Skip: {path}")
                continue
            if resize and image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            writer.write(image)
    finally:
        writer.release()