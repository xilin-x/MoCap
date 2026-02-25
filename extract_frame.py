import os
import glob

import cv2


# load video and extract frames
def extract_frames(video_path, output_dir, verbose=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.transpose(frame, (1, 0, 2))
        frame = cv2.flip(frame, 0)
        if not ret:
            break

        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        if verbose:
            print(f"Saved {frame_filename}")
        frame_count += 1

    cap.release()
    if verbose:
        print(f"Extracted {frame_count} frames from {video_path} to {output_dir}")
    return frame_count

if __name__ == "__main__":
    seqs_name = sorted(os.listdir("/home/xilin/Projects/src/3D/sam-3d-body/data/videos"))
    for seq_name in seqs_name:
        videos_path = sorted(glob.glob(os.path.join("/home/xilin/Projects/src/3D/sam-3d-body/data/videos", seq_name, "*.MP4")))
        num_frame = [0, 0, 0]
        for idx, video_path in enumerate(videos_path):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join("results/frames", seq_name, video_name)

            num_frame[idx] += extract_frames(video_path, output_dir)

        print(f"Total frames extracted for {seq_name}: {num_frame}")
