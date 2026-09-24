"""Shared COCO-17 keypoint schema, used so drawing code doesn't need to know
which pose model produced the keypoints."""
from dataclasses import dataclass, field

COCO17_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class PersonPose:
    keypoints: list[Keypoint] = field(default_factory=list)  # len == 17, COCO17 order
    bbox: tuple[float, float, float, float] | None = None  # (x1, y1, x2, y2), optional
