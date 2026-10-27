"""Body measurements from one frame of the keypoints JSON.

Everything is measured in torso lengths (mid-shoulders to mid-hips), so the
numbers don't depend on how far the fighter is from the camera. The JSON
stores x as a fraction of the frame width and y as a fraction of its height;
`aspect` (width / height) puts both in the same unit before measuring.

A measurement is None when a keypoint it needs is missing or less confident
than config.MIN_KEYPOINT_CONF, or the fighter wasn't found.
"""
import math

from pipeline import config
from pipeline.common.keypoints import COCO17_NAMES

HANDS = ("left", "right")

Point = tuple[float, float]


def point(frame: dict, name: str, aspect: float) -> Point | None:
    """(x, y) of a keypoint in frame-height units, or None if not usable."""
    if not frame["found"]:
        return None
    x, y, conf = frame["keypoints"][COCO17_NAMES.index(name)]
    return (x * aspect, y) if conf >= config.MIN_KEYPOINT_CONF else None


def _dist(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _mid(a: Point, b: Point) -> Point:
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def torso_length(frame: dict, aspect: float) -> float | None:
    """Mid-shoulders to mid-hips: the unit every measurement is given in."""
    ls, rs, lh, rh = (point(frame, n, aspect) for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    if None in (ls, rs, lh, rh):
        return None
    length = _dist(_mid(ls, rs), _mid(lh, rh))
    return length if length > 0 else None


def guard_height(frame: dict, aspect: float, hand: str) -> float | None:
    """How far the wrist is above its shoulder, in torso lengths: 0 = level,
    negative = below. A tight guard still reads below 0, as the wrist point
    sits low inside the glove (DECISIONS.md 011)."""
    torso = torso_length(frame, aspect)
    shoulder, wrist = point(frame, f"{hand}_shoulder", aspect), point(frame, f"{hand}_wrist", aspect)
    if torso is None or shoulder is None or wrist is None:
        return None
    return (shoulder[1] - wrist[1]) / torso  # image y grows downwards


def arm_extension(frame: dict, aspect: float, hand: str) -> float | None:
    """Shoulder-to-wrist distance on screen, in torso lengths."""
    torso = torso_length(frame, aspect)
    shoulder, wrist = point(frame, f"{hand}_shoulder", aspect), point(frame, f"{hand}_wrist", aspect)
    if torso is None or shoulder is None or wrist is None:
        return None
    return _dist(shoulder, wrist) / torso


def elbow_angle(frame: dict, aspect: float, hand: str) -> float | None:
    """Angle at the elbow between upper arm and forearm, in degrees on screen:
    180 = arm straight. A punch thrown towards the camera can look bent."""
    s, e, w = (point(frame, f"{hand}_{part}", aspect) for part in ("shoulder", "elbow", "wrist"))
    if s is None or e is None or w is None:
        return None
    upper, fore = _dist(e, s), _dist(e, w)
    if upper == 0 or fore == 0:
        return None
    cos = ((s[0] - e[0]) * (w[0] - e[0]) + (s[1] - e[1]) * (w[1] - e[1])) / (upper * fore)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def wrist_offset(frame: dict, aspect: float, hand: str) -> Point | None:
    """Wrist position relative to its shoulder, in torso lengths, so moving
    the whole body (stepping in) doesn't move it."""
    torso = torso_length(frame, aspect)
    shoulder, wrist = point(frame, f"{hand}_shoulder", aspect), point(frame, f"{hand}_wrist", aspect)
    if torso is None or shoulder is None or wrist is None:
        return None
    return (wrist[0] - shoulder[0]) / torso, (wrist[1] - shoulder[1]) / torso


def facing(frame: dict, aspect: float) -> float | None:
    """Shoulder width / torso length: how square the fighter stands to the
    camera. Roughly under 0.35 side-on, 0.35-0.6 angled, 0.6+ facing the
    camera (DECISIONS.md 012)."""
    torso = torso_length(frame, aspect)
    ls, rs = point(frame, "left_shoulder", aspect), point(frame, "right_shoulder", aspect)
    if torso is None or ls is None or rs is None:
        return None
    return _dist(ls, rs) / torso
