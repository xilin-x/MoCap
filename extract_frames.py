import os
import glob
import cv2
import argparse
from concurrent.futures import ThreadPoolExecutor

from src.script_utils import iter_sequences


def extract_frames(video_path, output_dir, workers=8, resume=False):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    existing = set()

    if resume:
        for path in glob.glob(os.path.join(output_dir, "frame_*.jpg")):
            name = os.path.splitext(os.path.basename(path))[0]
            existing.add(int(name.split("_")[-1]))

        start = next((i for i in range(total) if i not in existing), total)

        if start >= total:
            cap.release()
            return total

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    else:
        start = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        idx = start

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx not in existing:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                path = os.path.join(output_dir, f"frame_{idx:04d}.jpg")

                pool.submit(cv2.imwrite, path, frame)

            idx += 1

    cap.release()
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-w", "--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for seq_name, seq_dir in iter_sequences(args.input):
        videos = sorted(glob.glob(os.path.join(seq_dir, "*.MP4")))

        counts = []

        for video_path in videos:
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            output_dir = os.path.join(args.output, seq_name, video_name)

            counts.append(extract_frames(video_path, output_dir, args.workers, args.resume))

        print(seq_name, counts)
