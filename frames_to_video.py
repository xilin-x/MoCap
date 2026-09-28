import argparse
from pathlib import Path

from src.video_utils import write_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    frames = sorted(args.input.glob("frame_*.ppm"))

    if not frames:
        raise RuntimeError(f"No frames found: {args.input}")

    write_video(
        frames,
        args.output,
        args.fps,
        progress=lambda index, path: print(
            f"[{index + 1}/{len(frames)}] {path.name}", flush=True
        ),
    )

    print(f"Saved: {args.output}")