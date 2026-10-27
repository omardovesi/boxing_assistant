"""Per-frame punch signals for one hand over a whole clip (DECISIONS.md 012).

Each function takes the JSON's frames and returns one value per frame (None
where it can't be measured), so frame numbers stay in step with the video.
Values are median-filtered first (config.SMOOTH_SECONDS) to drop one-frame
keypoint jumps.

- extension_above_baseline: arm length beyond that hand's usual length, so an
  arm held out in a long guard isn't a punch but a jab from it still is.
- wrist_speed: how fast the wrist moves relative to its shoulder, so stepping
  in isn't a punch.
- elbow_opening: how many degrees the elbow straightened in the last moment.
"""
import statistics

from pipeline import config
from pipeline.body import arm_extension, elbow_angle, wrist_offset
from pipeline.metrics import frame_weight, median_filter


def _span_frames(fps: float, seconds: float) -> int:
    return max(1, round(seconds * fps))


def baseline(values: list[float | None], weights: list[float]) -> float | None:
    """Median over frames trusted enough for events: the hand's usual value."""
    usable = [v for v, w in zip(values, weights) if v is not None and w >= config.MIN_EVENT_WEIGHT]
    return statistics.median(usable) if usable else None


def extension_above_baseline(frames: list[dict], aspect: float, fps: float, hand: str) -> list[float | None]:
    ext = median_filter([arm_extension(f, aspect, hand) for f in frames], fps)
    base = baseline(ext, [frame_weight(f) for f in frames])
    return [None if v is None or base is None else v - base for v in ext]


def wrist_speed(frames: list[dict], aspect: float, fps: float, hand: str,
                span: float = config.SPEED_SPAN_SECONDS) -> list[float | None]:
    """Torso lengths per second, wrist relative to shoulder, over `span`."""
    offsets = [wrist_offset(f, aspect, hand) for f in frames]
    xs = median_filter([o and o[0] for o in offsets], fps)
    ys = median_filter([o and o[1] for o in offsets], fps)
    k = _span_frames(fps, span)
    out: list[float | None] = [None] * len(frames)
    for i in range(k, len(frames)):
        if None not in (xs[i], ys[i], xs[i - k], ys[i - k]):
            out[i] = ((xs[i] - xs[i - k]) ** 2 + (ys[i] - ys[i - k]) ** 2) ** 0.5 * fps / k
    return out


def elbow_opening(frames: list[dict], aspect: float, fps: float, hand: str,
                  span: float = config.OPENING_SPAN_SECONDS) -> list[float | None]:
    """Degrees the elbow opened: now minus the smallest angle in the last
    `span` seconds (0 when it's bending or steady)."""
    angles = median_filter([elbow_angle(f, aspect, hand) for f in frames], fps)
    k = _span_frames(fps, span)
    out: list[float | None] = [None] * len(frames)
    for i, a in enumerate(angles):
        before = [b for b in angles[max(0, i - k):i + 1] if b is not None]
        if a is not None and len(before) > 1:
            out[i] = a - min(before)
    return out


SIGNALS = {
    "extension": extension_above_baseline,
    "wrist_speed": wrist_speed,
    "elbow_opening": elbow_opening,
}
