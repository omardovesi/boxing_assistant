"""Guard-drop events for one fighter from the keypoints JSON (DECISIONS.md 011).

A guard drop = a hand stays clearly below *that fighter's own* usual guard
height (its median over trusted frames) for long enough, outside that hand's
punches and the moment just after one (the hand is still coming back). A
deliberately low guard (e.g. Philly shell) sets a low usual height, so it
only fires when the hand goes lower still; the fixed exposure threshold
(config.GUARD_EXPOSURE_THRESHOLD) is the measure of how low a guard is overall.
"""
from pipeline import config
from pipeline.body import HANDS, guard_height
from pipeline.metrics import frame_weight, median_filter
from pipeline.signals import baseline


def low_stretches(below: list[bool], fps: float, min_seconds: float = config.GUARD_DROP_SECONDS) -> list[tuple[int, int]]:
    """(first, last) frame of each unbroken run of True lasting at least
    `min_seconds`."""
    out, start = [], None
    for i, b in enumerate(below + [False]):
        if b and start is None:
            start = i
        elif not b and start is not None:
            if (i - start) / fps >= min_seconds:
                out.append((start, i - 1))
            start = None
    return out


def punch_zone(n_frames: int, fps: float, punches: list[dict],
               after: float = config.GUARD_DROP_AFTER_PUNCH) -> list[bool]:
    """True on frames from each punch's start to `after` seconds past its end."""
    zone = [False] * n_frames
    for p in punches:
        for i in range(max(0, round(p["start"] * fps)), min(n_frames, round((p["end"] + after) * fps) + 1)):
            zone[i] = True
    return zone


def detect_guard_drops(frames: list[dict], aspect: float, fps: float, hand: str, punches: list[dict],
                       usable: list[bool] | None = None, repeats: list[bool] | None = None) -> list[dict]:
    """Guard drops for one hand, in time order. `punches` are that hand's
    punch events; `repeats` (repeated pictures) makes the median filter run
    over distinct pictures, as for the guard threshold."""
    if usable is None:
        usable = [frame_weight(f) >= config.MIN_EVENT_WEIGHT for f in frames]
    guard = median_filter([guard_height(f, aspect, hand) for f in frames], fps, repeats=repeats)
    usual = baseline(guard, [1.0 if u else 0.0 for u in usable])
    if usual is None:
        return []
    limit = usual - config.GUARD_DROP_MARGIN
    zone = punch_zone(len(frames), fps, punches)
    below = [u and not z and g is not None and g < limit for g, u, z in zip(guard, usable, zone)]
    events = []
    for first, last in low_stretches(below, fps):
        lowest = min(guard[first:last + 1])
        events.append({
            "type": "guard_drop", "hand": hand,
            "start": round(first / fps, 3), "end": round((last + 1) / fps, 3),
            "duration": round((last + 1 - first) / fps, 3),
            "usual": round(usual, 2), "lowest": round(lowest, 2), "below_usual": round(usual - lowest, 2),
        })
    return events


def detect_all(data: dict, punches: list[dict], usable: list[bool] | None = None,
               repeats: list[bool] | None = None) -> list[dict]:
    """Both hands' guard drops from a loaded keypoints JSON, sorted by time.
    `punches` holds both hands' punch events; each hand only avoids its own."""
    aspect = data["width"] / data["height"]
    events = [e for hand in HANDS
              for e in detect_guard_drops(data["frames"], aspect, data["fps"], hand,
                                          [p for p in punches if p["hand"] == hand], usable, repeats)]
    return sorted(events, key=lambda e: e["start"])
