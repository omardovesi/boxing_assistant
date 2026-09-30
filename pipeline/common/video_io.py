"""Video segment extraction and annotated-video writing, via OpenCV.

No ffmpeg is available on this machine, so all video I/O goes through
cv2.VideoCapture / cv2.VideoWriter rather than shelling out.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoSegment:
    frames: list[np.ndarray]  # BGR frames, as read by OpenCV
    fps: float
    width: int
    height: int


class VideoStream:
    """Frames from `path` in [start, start + duration) seconds, decoded one at
    a time so memory stays flat however long the video is (a full 1440p clip
    decoded at once needs ~40-50 GB). `duration=None` means to the end.

    Metadata (fps, width, height, frame_count) is available as soon as it's
    opened. Use as a context manager, or call close(), to release the file.

    Raises FileNotFoundError if the video can't be opened, and ValueError if
    the requested window has no overlap with the video's actual length.
    """

    def __init__(self, path: str | Path, start: float = 0.0, duration: float | None = None):
        path = Path(path)
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"could not open video: {path}")

        try:
            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if self.fps <= 0:
                raise ValueError(f"video reports invalid fps ({self.fps}): {path}")

            start_frame = int(round(start * self.fps))
            if duration is None:
                end_frame = total_frames
            else:
                end_frame = int(round((start + duration) * self.fps))
                if total_frames > 0:
                    end_frame = min(end_frame, total_frames)
            if start_frame >= end_frame:
                end = "end" if duration is None else f"{start + duration}"
                raise ValueError(
                    f"requested segment [{start}, {end})s has no "
                    f"overlap with video of length ~{total_frames / self.fps:.1f}s"
                )

            # Expected number of frames; the container's count can be slightly
            # off, so iteration may stop a little early.
            self.frame_count = end_frame - start_frame
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        except Exception:
            self._cap.release()
            raise

    def __iter__(self) -> Iterator[np.ndarray]:
        for _ in range(self.frame_count):
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoStream":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_segment(path: str | Path, start: float, duration: float | None) -> VideoSegment:
    """Read all frames in [start, start + duration) seconds into memory. Fine
    for short segments; use VideoStream for full videos."""
    with VideoStream(path, start, duration) as stream:
        frames = list(stream)
        return VideoSegment(frames=frames, fps=stream.fps, width=stream.width, height=stream.height)


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
