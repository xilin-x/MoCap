import os
import glob
import argparse

from src.script_utils import iter_videos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-b", "--bboxes", required=True)
    parser.add_argument("-v", "--video", nargs="+", default=["all"])
    args = parser.parse_args()

    total_missing = 0

    for seq_name, video_name, frame_dir in iter_videos(args.input, args.video):
        bbox_dir = os.path.join(args.bboxes, seq_name, video_name)

        frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
        missing = []

        for frame in frames:
            name = os.path.splitext(os.path.basename(frame))[0]
            bbox_path = os.path.join(bbox_dir, f"{name}.npy")
            if not os.path.exists(bbox_path):
                missing.append(name)

        if not missing:
            print(f"OK      {seq_name}/{video_name}: {len(frames)}/{len(frames)}")
        else:
            total_missing += len(missing)
            print(f"MISSING {seq_name}/{video_name}: {len(missing)}/{len(frames)}")
            print("        " + ", ".join(missing))

    print()
    print(f"Total missing: {total_missing}")

    if total_missing > 0:
        raise SystemExit(1)
