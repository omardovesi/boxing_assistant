"""Runs YOLOv8-pose on a video segment and saves an annotated video.

DECISIONS.md item 005 resolved this as the pose model, after visually
comparing it against MediaPipe and RTMPose (see DECISIONS.md for why).
This script is still a standalone step, not the Phase 1 pipeline.analyze
tool — it draws every detected person every frame with no cross-frame
identity tracking.

Usage:
    python -m pipeline.run_pose_estimation --video "data/clip.mp4" \
        --start 0 --duration 10 --out outputs/pose_comparison/
"""
import argparse
import json
import time
from pathlib import Path

from pipeline.common.drawing import draw_skeleton
from pipeline.common.video_io import SegmentWriter, open_segment
from pipeline.models.yolov8_estimator import YOLOv8Estimator


def run(estimator: YOLOv8Estimator, segment, out_path: Path, conf_threshold: float) -> dict:
    print("loading model...")
    t0 = time.monotonic()
    estimator.load()
    load_s = time.monotonic() - t0

    print(f"running on {len(segment.frames)} frames...")
    t0 = time.monotonic()
    with SegmentWriter(out_path, segment.fps, segment.width, segment.height) as writer:
        for frame in segment.frames:
            people = estimator.predict(frame)
            annotated = draw_skeleton(frame, people, confidence_threshold=conf_threshold)
            writer.write(annotated)
    predict_s = time.monotonic() - t0

    fps = len(segment.frames) / predict_s if predict_s > 0 else float("inf")
    print(f"done -> {out_path} ({fps:.1f} fps inference, {load_s:.1f}s load)")
    return {"model": "yolov8", "load_seconds": load_s, "predict_seconds": predict_s, "inference_fps": fps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to a source video, e.g. data/clip.mp4")
    parser.add_argument("--start", type=float, default=0.0, help="Segment start time, in seconds")
    parser.add_argument("--duration", type=float, default=10.0, help="Segment duration, in seconds")
    parser.add_argument("--out", default="outputs/pose_comparison", help="Output directory")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="Detection/keypoint confidence threshold")
    parser.add_argument("--yolo-variant", default="s", help="YOLOv8-pose variant: n, s, m, l, x")
    parser.add_argument("--device", default="auto", help="Device for torch: auto, cpu, cuda")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "yolov8.mp4"

    print(f"Reading segment [{args.start}, {args.start + args.duration})s from {video_path}...")
    segment = open_segment(video_path, args.start, args.duration)
    print(f"Got {len(segment.frames)} frames at {segment.fps:.1f} fps, {segment.width}x{segment.height}")

    device = "cuda" if args.device == "auto" else args.device
    estimator = YOLOv8Estimator(variant=args.yolo_variant, conf_threshold=args.conf_threshold, device=device)
    stats = run(estimator, segment, out_path, args.conf_threshold)

    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps({"video": str(video_path), "start": args.start, "duration": args.duration, "runs": [stats]}, indent=2))
    print(f"\nWrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
