"""Metrics file for one fighter: summary numbers plus a time-sorted list of
punch and guard-drop events (Phase 2).

    python -m pipeline.summary outputs/analyze/clip/fighter_1.json [--repeats cache.json]

writes fighter_1_metrics.json next to the keypoints JSON. The keypoints file
stays as it is, so metrics can be recomputed any time with new settings.

Every number comes with its coverage (DECISIONS.md 010): how much trusted
footage it rests on. Guard height is median-filtered over distinct pictures
when it's known which frames repeat the one before (`--repeats`, e.g. the
labelling kit's cache, or a per-frame "repeat" flag in the JSON once
export.py writes one); otherwise over frames, and the file says which.
"""
import argparse
import json
from pathlib import Path

from pipeline import config
from pipeline.body import HANDS, guard_height
from pipeline.guard import detect_all as detect_guard_drops
from pipeline.metrics import frame_weight, median_filter, weight_summary, weighted_mean
from pipeline.punches import detect_all as detect_punches
from pipeline.signals import baseline


def _r(x: float | None, digits: int = 3) -> float | None:
    return None if x is None else round(x, digits)


def _result(r) -> dict:
    return {"value": _r(r.value), "coverage": _r(r.coverage)}


def frame_repeats(data: dict, repeats: list[bool] | None = None) -> list[bool] | None:
    """Which frames repeat the picture before: the JSON's own flags if every
    frame has one, else `repeats` if given, else None (unknown)."""
    frames = data["frames"]
    if frames and all("repeat" in f for f in frames):
        return [bool(f["repeat"]) for f in frames]
    return repeats


def build_summary(data: dict, repeats: list[bool] | None = None) -> dict:
    frames, fps = data["frames"], data["fps"]
    aspect = data["width"] / data["height"]
    repeats = frame_repeats(data, repeats)
    weights = [frame_weight(f) for f in frames]
    usable = [w >= config.MIN_EVENT_WEIGHT for w in weights]
    usable_minutes = sum(usable) / fps / 60

    punches = detect_punches(data, usable)
    drops = detect_guard_drops(data, punches, usable, repeats)

    guard: dict = {}
    frames_per_minute = round(60 * fps)
    for hand in HANDS:
        heights = median_filter([guard_height(f, aspect, hand) for f in frames], fps, repeats=repeats)
        below = [None if g is None else float(g < config.GUARD_EXPOSURE_THRESHOLD) for g in heights]
        guard[hand] = {
            "height": _result(weighted_mean(heights, weights)),
            "usual": _r(baseline(heights, weights)),
            "exposure": _result(weighted_mean(below, weights)),
            "drops": sum(e["hand"] == hand for e in drops),
            "height_per_minute": [
                {"minute": m, **_result(weighted_mean(heights[i:i + frames_per_minute], weights[i:i + frames_per_minute]))}
                for m, i in enumerate(range(0, len(frames), frames_per_minute))],
        }

    events = [{"time": e["peak"], **e} for e in punches] + [{"time": e["start"], **e} for e in drops]
    events.sort(key=lambda e: (e["time"], e["type"], e["hand"]))
    return {
        "source": data.get("video"),
        "fighter_id": data.get("fighter_id"),
        "fps": fps,
        "duration_s": _r(len(frames) / fps),
        "usable_minutes": _r(usable_minutes),
        "frames": {k: _r(v) for k, v in weight_summary(frames).items()},
        "clinches": len(data.get("clinches", [])),
        "guard_filter": "pictures" if repeats is not None else "frames",
        "settings": {
            "guard_exposure_threshold": config.GUARD_EXPOSURE_THRESHOLD,
            "guard_drop_margin": config.GUARD_DROP_MARGIN,
            "guard_drop_seconds": config.GUARD_DROP_SECONDS,
            "punch_speed": config.PUNCH_SPEED,
            "punch_extension": config.PUNCH_EXTENSION,
            "min_event_weight": config.MIN_EVENT_WEIGHT,
        },
        "guard": guard,
        "punches": {
            "total": len(punches),
            "left": sum(e["hand"] == "left" for e in punches),
            "right": sum(e["hand"] == "right" for e in punches),
            "per_usable_minute": _r(len(punches) / usable_minutes, 2) if usable_minutes else None,
        },
        "events": events,
    }


def output_path(keypoints_json: Path) -> Path:
    return keypoints_json.with_name(f"{keypoints_json.stem}_metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json", type=Path, help="Keypoints JSON written by pipeline.analyze")
    parser.add_argument("--repeats", type=Path, help="JSON list of per-frame repeat flags (e.g. the labelling kit cache)")
    args = parser.parse_args()

    data = json.loads(args.json.read_text(encoding="utf-8"))
    repeats = json.loads(args.repeats.read_text(encoding="utf-8")) if args.repeats else None
    summary = build_summary(data, repeats)
    out = output_path(args.json)
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    g, p = summary["guard"], summary["punches"]
    print(f"{out}: {summary['usable_minutes']} usable minutes; {p['total']} punches ({p['per_usable_minute']} per minute);"
          f" exposure left {g['left']['exposure']['value']}, right {g['right']['exposure']['value']};"
          f" {g['left']['drops'] + g['right']['drops']} guard drops; {len(summary['events'])} events")


if __name__ == "__main__":
    main()
