import cv2
import numpy as np
import pytest

from pipeline.common.video_io import SegmentWriter, open_segment

FPS = 10.0
WIDTH, HEIGHT = 64, 48
TOTAL_FRAMES = 20  # 2 seconds at 10fps


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    for i in range(TOTAL_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_open_segment_reads_expected_frame_count(synthetic_video):
    # 1 second of a 2-second, 10fps video -> ~10 frames
    segment = open_segment(synthetic_video, start=0, duration=1.0)
    assert segment.fps == pytest.approx(FPS, rel=0.05)
    assert segment.width == WIDTH
    assert segment.height == HEIGHT
    assert len(segment.frames) == pytest.approx(10, abs=1)


def test_open_segment_clamps_to_video_length(synthetic_video):
    # Requesting far more than the video's length should just return what exists.
    segment = open_segment(synthetic_video, start=0, duration=100.0)
    assert len(segment.frames) <= TOTAL_FRAMES


def test_open_segment_respects_start_offset(synthetic_video):
    full = open_segment(synthetic_video, start=0, duration=2.0)
    offset = open_segment(synthetic_video, start=1.0, duration=1.0)
    assert len(offset.frames) < len(full.frames)


def test_open_segment_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_segment(tmp_path / "does_not_exist.mp4", start=0, duration=1.0)


def test_open_segment_raises_when_start_past_end(synthetic_video):
    with pytest.raises(ValueError):
        open_segment(synthetic_video, start=10.0, duration=1.0)


def test_segment_writer_output_is_reopenable(tmp_path):
    out_path = tmp_path / "out" / "written.mp4"
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    with SegmentWriter(out_path, fps=FPS, width=WIDTH, height=HEIGHT) as writer:
        for _ in range(5):
            writer.write(frame)

    assert out_path.exists()
    cap = cv2.VideoCapture(str(out_path))
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count == 5
