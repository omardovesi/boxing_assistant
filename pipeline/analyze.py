"""Phase 1 CLI: track every person in a video and draw one selected fighter.

Step 1 — preview the first frame with numbered fighters, to pick an ID:
    python -m pipeline.analyze "data/clip.mp4" --preview

Step 2 — draw only that fighter's skeleton for the whole segment:
    python -m pipeline.analyze "data/clip.mp4" --fighter 1

Use the same --start for both steps: track IDs are assigned from the first
frame of the segment, so a different start frame can produce different IDs.
"""
import argparse
import time
from pathlib import Path

import cv2

from pipeline.common.drawing import draw_id_labels, draw_skeleton
from pipeline.common.video_io import SegmentWriter, open_segment
from pipeline.models.yolov8_estimator import YOLOv8Estimator
from pipeline.tracking import select_fighter, track_ids


def preview(estimator: YOLOv8Estimator, segment, out_dir: Path) -> None:
    people = estimator.track(segment.frames[0])
    out_path = out_dir / "preview.jpg"
    cv2.imwrite(str(out_path), draw_id_labels(segment.frames[0], people))

    ids = track_ids(people)
    if ids:
        print(f"Fighters in first frame: {ids}")
        print(f"Open {out_path} to see who is who, then rerun with --fighter ID")
    else:
        print(f"No people detected in the first frame. See {out_path}")


def analyze(estimator: YOLOv8Estimator, segment, fighter_id: int, out_dir: Path, conf_threshold: float) -> None:
    out_path = out_dir / f"fighter_{fighter_id}.mp4"
    missing = 0

    print(f"Tracking fighter {fighter_id} over {len(segment.frames)} frames...")
    t0 = time.monotonic()
    with SegmentWriter(out_path, segment.fps, segment.width, segment.height) as writer:
        for frame in segment.frames:
            fighter = select_fighter(estimator.track(frame), fighter_id)
            if fighter is None:
                missing += 1
                writer.write(frame)
            else:
                writer.write(draw_skeleton(frame, [fighter], confidence_threshold=conf_threshold))
    elapsed = time.monotonic() - t0

    n = len(segment.frames)
    print(f"done -> {out_path} ({n / elapsed:.1f} fps)")
    if missing:
        print(f"warning: fighter {fighter_id} not found in {missing}/{n} frames ({missing / n:.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to a source video, e.g. data/clip.mp4")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true", help="Save first frame with numbered fighters")
    mode.add_argument("--fighter", type=int, help="Track ID of the fighter to analyze (from --preview)")
    parser.add_argument("--start", type=float, default=0.0, help="Segment start time, in seconds")
    parser.add_argument("--duration", type=float, default=10.0, help="Segment duration, in seconds")
    parser.add_argument("--out", default="outputs/analyze", help="Output directory")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="Detection/keypoint confidence threshold")
    parser.add_argument("--device", default="auto", help="Device for torch: auto, cpu, cuda")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preview only needs the first frame; skip decoding the rest.
    duration = 1.0 if args.preview else args.duration
    segment = open_segment(video_path, args.start, duration)
    print(f"Read {len(segment.frames)} frames at {segment.fps:.1f} fps, {segment.width}x{segment.height}")

    estimator = YOLOv8Estimator(conf_threshold=args.conf_threshold, device=args.device)
    estimator.load()

    if args.preview:
        preview(estimator, segment, out_dir)
    else:
        analyze(estimator, segment, args.fighter, out_dir, args.conf_threshold)


if __name__ == "__main__":
    main()
