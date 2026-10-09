"""Per-frame keypoints JSON for the followed fighter, for Phase 2 metrics
without re-running the pose model.

Coordinates stay normalized 0-1 as the pose model returns them; the file's
width/height convert them to pixels. Each keypoint is [x, y, confidence] in
COCO-17 order, with the names listed once at the top of the file.

Each frame also stores `frame_weight` (how much metrics should trust it, see
pipeline/metrics.py) and the raw inputs it's computed from (`overlap`,
`body_confidence`, the keypoint confidences), so weights can be recomputed
with different limits without re-running the video.
"""
import json
from pathlib import Path

from pipeline.common.keypoints import PersonPose
from pipeline.metrics import body_confidence, frame_weight
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


def frame_record(frame_idx: int, time: float, fighter: PersonPose | None, people: list[PersonPose],
                 in_clinch: bool = False) -> dict:
    """One frame's entry. Missing frames keep their entry so frame numbers
    and times always line up with the video. A missing frame during a
    clinch (identity not assigned) is marked `"clinch": true`."""
    record = {"frame": frame_idx, "time": _r(time), "found": fighter is not None}
    if fighter is None:
        if in_clinch:
            record["clinch"] = True
        record["frame_weight"] = 0.0
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
    # From the rounded stored values, so recomputing from the file matches.
    record["body_confidence"] = _r(body_confidence(record["keypoints"]))
    record["frame_weight"] = _r(frame_weight(record))
    return record


def write_keypoints_json(path: Path, meta: dict, frames: list[dict]) -> None:
    """Save `meta` fields followed by the per-frame entries."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**meta, "frames": frames}, f, separators=(",", ":"))
