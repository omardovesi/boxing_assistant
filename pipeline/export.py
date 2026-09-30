"""Per-frame keypoints JSON for the followed fighter, for Phase 2 metrics
without re-running the pose model.

Coordinates stay normalized 0-1 as the pose model returns them; the file's
width/height convert them to pixels. Each keypoint is [x, y, confidence] in
COCO-17 order, with the names listed once at the top of the file.
"""
import json
from pathlib import Path

from pipeline.common.keypoints import PersonPose
from pipeline.tracking import bbox_iou, nearest_opponent

# Box overlap with the opponent at which a frame counts as a clinch. The raw
# overlap is stored too, so Phase 2 can change this without re-running video.
CLINCH_IOU = 0.3
DECIMALS = 4  # under 1 pixel at 1440p


def _r(value: float) -> float:
    return round(float(value), DECIMALS)


def clinch_overlap(fighter: PersonPose, people: list[PersonPose]) -> float:
    """IoU of the fighter's box with the nearest person of similar size:
    0 = apart (or nobody nearby), 1 = same box."""
    if fighter.bbox is None:
        return 0.0
    opponent = nearest_opponent(people, fighter)
    if opponent is None:
        return 0.0
    return bbox_iou(fighter.bbox, opponent.bbox)


def frame_record(frame_idx: int, time: float, fighter: PersonPose | None, people: list[PersonPose]) -> dict:
    """One frame's entry. Missing frames keep their entry so frame numbers
    and times always line up with the video."""
    record = {"frame": frame_idx, "time": _r(time), "found": fighter is not None}
    if fighter is None:
        return record
    # Round first so `clinch` always agrees with the stored `overlap`.
    overlap = _r(clinch_overlap(fighter, people))
    record.update({
        "track_id": fighter.track_id,
        "bbox": None if fighter.bbox is None else [_r(v) for v in fighter.bbox],
        "keypoints": [[_r(k.x), _r(k.y), _r(k.confidence)] for k in fighter.keypoints],
        "overlap": overlap,
        "clinch": overlap >= CLINCH_IOU,
    })
    return record


def write_keypoints_json(path: Path, meta: dict, frames: list[dict]) -> None:
    """Save `meta` fields followed by the per-frame entries."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**meta, "frames": frames}, f, separators=(",", ":"))
