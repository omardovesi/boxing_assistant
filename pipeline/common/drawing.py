"""Draws a COCO-17 skeleton over a frame, shared across all three pose models
so their outputs are visually comparable."""
import cv2
import numpy as np

from pipeline.common.keypoints import COCO17_NAMES, PersonPose

_NAME_TO_INDEX = {name: i for i, name in enumerate(COCO17_NAMES)}

SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
]
SKELETON_EDGE_INDICES = [(_NAME_TO_INDEX[a], _NAME_TO_INDEX[b]) for a, b in SKELETON_EDGES]

_PERSON_COLORS = [
    (0, 255, 0),    # green
    (0, 128, 255),  # orange
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (0, 0, 255),    # red
    (255, 0, 0),    # blue
]


def draw_skeleton(
    frame: np.ndarray,
    people: list[PersonPose],
    confidence_threshold: float = 0.3,
) -> np.ndarray:
    """Returns a copy of `frame` with each person's COCO-17 skeleton drawn on
    it. Keypoints/edges below `confidence_threshold` are skipped. Assumes
    keypoint x/y are normalized to [0, 1]; scales to the frame's pixel size.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    for person_idx, person in enumerate(people):
        color = _PERSON_COLORS[person_idx % len(_PERSON_COLORS)]
        kps = person.keypoints

        for i, j in SKELETON_EDGE_INDICES:
            if i >= len(kps) or j >= len(kps):
                continue
            a, b = kps[i], kps[j]
            if a.confidence < confidence_threshold or b.confidence < confidence_threshold:
                continue
            pt_a = (int(a.x * w), int(a.y * h))
            pt_b = (int(b.x * w), int(b.y * h))
            cv2.line(out, pt_a, pt_b, color, 2)

        for kp in kps:
            if kp.confidence < confidence_threshold:
                continue
            pt = (int(kp.x * w), int(kp.y * h))
            cv2.circle(out, pt, 3, color, -1)

    return out
