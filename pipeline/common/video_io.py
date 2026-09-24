"""Video segment extraction and annotated-video writing, via OpenCV.

No ffmpeg is available on this machine, so all video I/O goes through
cv2.VideoCapture / cv2.VideoWriter rather than shelling out.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoSegment:
    frames: list[np.ndarray]  # BGR frames, as read by OpenCV
    fps: float
    width: int
    height: int


def open_segment(path: str | Path, start: float, duration: float) -> VideoSegment:
    """Read frames from `path` in [start, start + duration) seconds.

    Raises FileNotFoundError if the video can't be opened, and ValueError if
    the requested window has no overlap with the video's actual length.
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            raise ValueError(f"video reports invalid fps ({fps}): {path}")

        start_frame = int(round(start * fps))
        end_frame = int(round((start + duration) * fps))
        if total_frames > 0:
            end_frame = min(end_frame, total_frames)
        if start_frame >= end_frame:
            raise ValueError(
                f"requested segment [{start}, {start + duration})s has no "
                f"overlap with video of length ~{total_frames / fps:.1f}s"
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(end_frame - start_frame):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)

        return VideoSegment(frames=frames, fps=fps, width=width, height=height)
    finally:
        cap.release()


class SegmentWriter:
    """Thin wrapper around cv2.VideoWriter for writing an annotated segment."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open video writer for: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "SegmentWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
