from pipeline.common.keypoints import Keypoint

# Rough standing pose as (x, y) fractions of the person's box, in COCO-17 order.
_POSE = [
    (0.50, 0.10),                # nose
    (0.45, 0.07), (0.55, 0.07),  # eyes
    (0.35, 0.09), (0.65, 0.09),  # ears
    (0.20, 0.25), (0.80, 0.25),  # shoulders
    (0.15, 0.40), (0.85, 0.40),  # elbows
    (0.20, 0.50), (0.80, 0.50),  # wrists
    (0.30, 0.55), (0.70, 0.55),  # hips
    (0.30, 0.75), (0.70, 0.75),  # knees
    (0.30, 0.95), (0.70, 0.95),  # ankles
]


def paint_kit(frame, person, head=None, shirt=None, trunks=None) -> None:
    """Paint a person's kit as solid blocks (BGR) on `frame`, in place:
    headgear around/above the top of the box, shirt, then trunks + legs."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = person.bbox
    bw, bh = x2 - x1, y2 - y1
    blocks = [
        (head, x1 - 0.2 * bw, y1 - 0.15 * bh, x2 + 0.2 * bw, y1 + 0.15 * bh),
        (shirt, x1, y1 + 0.15 * bh, x2, y1 + 0.55 * bh),
        (trunks, x1, y1 + 0.55 * bh, x2, y2),
    ]
    for colour, bx1, by1, bx2, by2 in blocks:
        if colour is not None:
            frame[max(0, int(by1 * h)):int(by2 * h), max(0, int(bx1 * w)):int(bx2 * w)] = colour


def body_keypoints(bbox, confidence: float = 0.9) -> list[Keypoint]:
    """17 confident keypoints laid out inside `bbox` (normalized x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return [Keypoint(x=x1 + fx * (x2 - x1), y=y1 + fy * (y2 - y1), confidence=confidence) for fx, fy in _POSE]
