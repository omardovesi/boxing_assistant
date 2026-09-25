"""Draws COCO-17 skeletons (and, for fighter selection, numbered track-ID
boxes) over a frame."""
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
        color = _person_color(person, person_idx)
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


def draw_id_labels(frame: np.ndarray, people: list[PersonPose]) -> np.ndarray:
    """Returns a copy of `frame` with each tracked person's bbox and a large
    track ID number, so the user can choose which fighter to analyze. People
    without a bbox or track_id are skipped."""
    out = frame.copy()
    h, w = out.shape[:2]
    # Scale label size with resolution so IDs stay readable on 720p through 4K.
    scale = max(h, w) / 1000
    font_scale, thickness = 1.5 * scale, max(2, int(4 * scale))

    for person_idx, person in enumerate(people):
        if person.bbox is None or person.track_id is None:
            continue
        color = _person_color(person, person_idx)
        x1, y1, x2, y2 = person.bbox
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(out, p1, p2, color, thickness)
        label_pos = (p1[0] + thickness, p1[1] + int(45 * scale))  # inside the box, top-left
        cv2.putText(out, str(person.track_id), label_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return out


def _person_color(person: PersonPose, person_idx: int) -> tuple[int, int, int]:
    # Prefer track_id so a person keeps the same color across frames.
    key = person.track_id if person.track_id is not None else person_idx
    return _PERSON_COLORS[key % len(_PERSON_COLORS)]
