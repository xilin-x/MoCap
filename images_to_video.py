import argparse
from pathlib import Path

from rich.progress import track
from src.video_utils import write_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for seq_dir in sorted(path for path in args.input.iterdir() if path.is_dir()):
        for image_dir in sorted(path for path in seq_dir.iterdir() if path.is_dir()):
            images = sorted(image_dir.glob("frame_*.jpg"))
            output = args.output / seq_dir.name / f"{image_dir.name}.mp4"
            if not images:
                continue
            if args.resume and output.exists():
                print(f"SKIP: {output}")
                continue

            progress = track(images, description=f"{seq_dir.name}/{image_dir.name}")
            try:
                write_video(
                    images,
                    output,
                    args.fps,
                    progress=lambda *_: next(progress),
                    resize=True,
                )
            finally:
                progress.close()
            print(f"Saved: {output}")
