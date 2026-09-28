import os
import sys

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn


def setup_paths(root):
    for path in ("lib", "lib/body_model", "lib/sam3"):
        path = os.path.join(root, path)
        if path not in sys.path:
            sys.path.insert(0, path)


def setup_torch():
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def iter_sequences(input_dir):
    for seq_name in sorted(os.listdir(input_dir)):
        seq_dir = os.path.join(input_dir, seq_name)
        if os.path.isdir(seq_dir):
            yield seq_name, seq_dir


def iter_videos(input_dir, selected):
    for seq_name, seq_dir in iter_sequences(input_dir):
        for video_name in sorted(os.listdir(seq_dir)):
            if "all" not in selected and video_name not in selected:
                continue
            frame_dir = os.path.join(seq_dir, video_name)
            if os.path.isdir(frame_dir):
                yield seq_name, video_name, frame_dir


def progress_bar():
    return Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )