"""Punch events for one fighter from the keypoints JSON (DECISIONS.md 012).

A punch = the wrist moves fast relative to its shoulder (wrist_speed above
config.PUNCH_SPEED) and around that moment the arm reaches beyond its usual
length (extension above config.PUNCH_EXTENSION). A wrist going out and coming
back gives two fast stretches close together; they're merged into one punch.

Only frames trusted enough for counts (frame weight >= MIN_EVENT_WEIGHT,
DECISIONS.md 010) can start a punch. Every event records how the fighter
faces the camera, so accuracy can be reported per view later.
"""
from pipeline import config
from pipeline.body import HANDS, facing
from pipeline.metrics import frame_weight
from pipeline.signals import extension_above_baseline, wrist_speed

FACE = (0, 1, 2)  # nose and eyes


def fast_stretches(speed: list[float | None], usable: list[bool], fps: float,
                   threshold: float = config.PUNCH_SPEED,
                   merge: float = config.PUNCH_MERGE_SECONDS) -> list[tuple[int, int]]:
    """(first, last) frame of each stretch where the wrist is faster than
    `threshold` on usable frames; stretches with a gap under `merge` seconds
    become one."""
    out: list[tuple[int, int]] = []
    for i, (v, ok) in enumerate(zip(speed, usable)):
        if not ok or v is None or v <= threshold:
            continue
        if out and (i - out[-1][1]) / fps < merge:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def detect_punches(frames: list[dict], aspect: float, fps: float, hand: str,
                   usable: list[bool] | None = None) -> list[dict]:
    """Punch events for one hand, in time order."""
    if usable is None:
        usable = [frame_weight(f) >= config.MIN_EVENT_WEIGHT for f in frames]
    speed = wrist_speed(frames, aspect, fps, hand)
    ext = extension_above_baseline(frames, aspect, fps, hand)
    reach = round(config.PUNCH_EXTENSION_SECONDS * fps)
    events = []
    for first, last in fast_stretches(speed, usable, fps):
        around = [ext[i] for i in range(max(0, first - reach), min(len(frames), last + reach + 1))
                  if usable[i] and ext[i] is not None]
        if not around or max(around) <= config.PUNCH_EXTENSION:
            continue
        peak = max(range(first, last + 1), key=lambda i: speed[i] if speed[i] is not None else -1.0)
        frame = frames[peak]
        events.append({
            "type": "punch", "hand": hand,
            "start": round(first / fps, 3), "peak": round(peak / fps, 3), "end": round(last / fps, 3),
            "peak_frame": peak, "peak_speed": round(speed[peak], 2), "extension": round(max(around), 2),
            "facing": None if (f := facing(frame, aspect)) is None else round(f, 2),
            "face_confidence": round(sum(frame["keypoints"][i][2] for i in FACE) / len(FACE), 2),
        })
    return events


def detect_all(data: dict, usable: list[bool] | None = None) -> list[dict]:
    """Both hands' punches from a loaded keypoints JSON, sorted by time."""
    aspect = data["width"] / data["height"]
    events = [e for hand in HANDS for e in detect_punches(data["frames"], aspect, data["fps"], hand, usable)]
    return sorted(events, key=lambda e: e["peak"])
