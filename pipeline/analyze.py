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
from pipeline.common.video_io import SegmentWriter, VideoStream, open_segment
from pipeline.models.yolov8_estimator import YOLOv8Estimator
from pipeline.tracking import FighterFollower, track_ids

# One colour for the followed fighter whatever their current track ID, so the
# skeleton doesn't change colour on every relink and a jump to the opponent is
# easy to spot. Cyan (BGR): unlikely to match any kit.
FIGHTER_COLOR = (255, 255, 0)


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


def analyze(estimator: YOLOv8Estimator, stream: VideoStream, fighter_id: int, out_dir: Path,
            conf_threshold: float, relink: bool = True, opponent_id: int | None = None) -> None:
    out_path = out_dir / f"fighter_{fighter_id}.mp4"
    missing = 0
    n = 0
    ids_seen: set[int] = set()
    follower = FighterFollower(fighter_id, aspect=stream.width / stream.height, relink=relink,
                               opponent_id=opponent_id)
    kit_announced = False
    progress_every = max(1, stream.frame_count // 10)

    print(f"Tracking fighter {fighter_id} over {stream.frame_count} frames...")
    t0 = time.monotonic()
    with SegmentWriter(out_path, stream.fps, stream.width, stream.height) as writer:
        for frame in stream:
            people = estimator.track(frame)
            ids_seen.update(track_ids(people))
            fighter = follower.update(people, frame)
            if not kit_announced and (follower.kit is not None or follower.kit_failed):
                kit_announced = True
                if follower.kit is not None:
                    print(f"  opponent {follower.opponent_id}; {follower.kit.describe()}", flush=True)
                else:
                    print("  warning: no kit item separates the fighters; following by track ID only")
            if fighter is None:
                missing += 1
                writer.write(frame)
            else:
                writer.write(draw_skeleton(frame, [fighter], confidence_threshold=conf_threshold,
                                           color=FIGHTER_COLOR))
            n += 1
            if n % progress_every == 0:
                print(f"  {n}/{stream.frame_count} frames ({n / stream.frame_count:.0%}), "
                      f"{n / (time.monotonic() - t0):.1f} fps", flush=True)
    elapsed = time.monotonic() - t0

    print(f"done -> {out_path} ({n / elapsed:.1f} fps)")
    # A new ID usually means the tracker lost someone and re-found them, so for
    # a two-fighter clip anything above 2 points to ID switches.
    print(f"distinct track IDs seen: {len(ids_seen)} {sorted(ids_seen)}")
    if follower.kit is None and not follower.kit_failed:
        print("warning: the two fighters were never apart long enough for a kit comparison; "
              "followed by track ID only")
    for frame_idx, old, new, kit_dist in follower.relinks:
        kit = "not checked" if kit_dist is None else f"{kit_dist:.2f}"
        print(f"relinked {old} -> {new} at frame {frame_idx} (kit distance {kit})")
    for frame_idx, old, new, old_dist, new_dist in follower.swaps:
        old_kit = "n/a" if old_dist is None else f"{old_dist:.2f}"
        print(f"swap corrected {old} -> {new} at frame {frame_idx} (kit distance {old_kit} -> {new_dist:.2f})")
    if missing:
        print(f"warning: fighter {fighter_id} not found in {missing}/{n} frames ({missing / n:.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to a source video, e.g. data/clip.mp4")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true", help="Save first frame with numbered fighters")
    mode.add_argument("--fighter", type=int, help="Track ID of the fighter to analyze (from --preview)")
    parser.add_argument("--start", type=float, default=0.0, help="Segment start time, in seconds")
    parser.add_argument("--duration", type=float, default=None,
                        help="Segment duration, in seconds (default: to the end of the video)")
    parser.add_argument("--out", default="outputs/analyze", help="Output directory")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="Detection/keypoint confidence threshold")
    parser.add_argument("--no-relink", action="store_true",
                        help="Don't take over a new track ID when the fighter's ID is lost")
    parser.add_argument("--opponent", type=int, default=None,
                        help="Track ID of the opponent (from --preview); default: the person nearest the fighter")
    parser.add_argument("--device", default="auto", help="Device for torch: auto, cpu, cuda")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    estimator = YOLOv8Estimator(conf_threshold=args.conf_threshold, device=args.device)
    estimator.load()

    if args.preview:
        # Preview only needs the first frame; skip decoding the rest.
        segment = open_segment(video_path, args.start, 1.0)
        preview(estimator, segment, out_dir)
    else:
        # Frames are streamed one at a time, so full-length videos fit in memory.
        with VideoStream(video_path, args.start, args.duration) as stream:
            print(f"Opened {stream.frame_count} frames at {stream.fps:.1f} fps, {stream.width}x{stream.height}")
            analyze(estimator, stream, args.fighter, out_dir, args.conf_threshold, relink=not args.no_relink,
                    opponent_id=args.opponent)


if __name__ == "__main__":
    main()
