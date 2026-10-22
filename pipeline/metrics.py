"""Frame weighting and the helpers every metric is built on.

Each frame of the keypoints JSON gets a weight in [0, 1]:

    frame_weight = overlap_factor * confidence_factor

- overlap_factor: 1 when the fighter is clear of the opponent, falling to 0 as
  their boxes overlap (borderline clinches count less, deep ones not at all).
- confidence_factor: 0 when the body keypoints are unreliable, rising to 1.
- 0 when the fighter wasn't found (no skeleton).

Averages are weighted by it, event counts ignore light frames, and every
metric reports its coverage (sum of weights / frames in the window), so a
number resting on little evidence is visible as such.

Show a clip's weights:
    python -m pipeline.metrics outputs/analyze/clip/fighter_1.json --frame 2617
"""
import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pipeline import config
from pipeline.common.keypoints import COCO17_NAMES

# Shoulders, elbows, wrists, hips, knees, ankles. The face points are left out:
# they're often low-confidence behind headgear, which says little about the body.
BODY = slice(COCO17_NAMES.index("left_shoulder"), COCO17_NAMES.index("right_ankle") + 1)


@dataclass
class MetricResult:
    value: float | None
    coverage: float  # sum of frame weights / frames in the window, 0-1


def ramp(value: float, zero_at: float, full_at: float) -> float:
    """0 at `zero_at`, 1 at `full_at`, linear in between and clamped outside.
    Works in either direction (full_at may be below zero_at)."""
    return min(1.0, max(0.0, (value - zero_at) / (full_at - zero_at)))


def overlap_factor(overlap: float) -> float:
    return ramp(overlap, config.WEIGHT_OVERLAP_ZERO, config.WEIGHT_OVERLAP_FULL)


def confidence_factor(body_conf: float) -> float:
    return ramp(body_conf, config.WEIGHT_CONF_ZERO, config.WEIGHT_CONF_FULL)


def body_confidence(keypoints: Sequence[Sequence[float]]) -> float:
    """Mean confidence of the 12 body keypoints, from [x, y, conf] triples."""
    body = keypoints[BODY]
    return sum(k[2] for k in body) / len(body)


def frame_weight(record: dict) -> float:
    """Weight of one JSON frame entry. Recomputed from the stored overlap and
    keypoints, so changes to config apply to files written earlier."""
    if not record["found"]:
        return 0.0
    return overlap_factor(record["overlap"]) * confidence_factor(body_confidence(record["keypoints"]))


def _coverage(weights: Sequence[float]) -> float:
    return sum(weights) / len(weights) if weights else 0.0


def weighted_mean(values: Sequence[float | None], weights: Sequence[float]) -> MetricResult:
    """Weighted average over a window. Frames without a value (None) count as
    weight 0 but still count towards the window's length."""
    effective = [0.0 if v is None else w for v, w in zip(values, weights, strict=True)]
    total = sum(effective)
    value = sum(v * w for v, w in zip(values, effective) if v is not None) / total if total > 0 else None
    return MetricResult(value, _coverage(effective))


def count_events(event_flags: Sequence[bool], weights: Sequence[float]) -> MetricResult:
    """Number of frames with an event, ignoring frames lighter than
    MIN_EVENT_WEIGHT."""
    count = sum(1 for flag, w in zip(event_flags, weights, strict=True) if flag and w >= config.MIN_EVENT_WEIGHT)
    return MetricResult(float(count), _coverage(weights))


def filter_frames(fps: float, seconds: float = config.SMOOTH_SECONDS) -> int:
    """Median filter length in frames for `seconds`: odd, at least 1."""
    n = max(1, round(seconds * fps))
    return n if n % 2 else n + 1


def median_filter(values: Sequence[float | None], fps: float,
                  seconds: float = config.SMOOTH_SECONDS) -> list[float | None]:
    """Running median over `seconds` (config.SMOOTH_SECONDS by default), to
    remove one-frame keypoint jumps. Missing values (None) stay missing and
    split the series: a value is only smoothed with neighbours from its own
    unbroken stretch. Near the end of a stretch the window shrinks evenly on
    both sides, so the first and last values are kept as they are."""
    half = filter_frames(fps, seconds) // 2
    out: list[float | None] = list(values)
    i = 0
    while i < len(values):
        if values[i] is None:
            i += 1
            continue
        start = i
        while i < len(values) and values[i] is not None:
            i += 1
        for j in range(start, i):  # stretch values[start:i] has no gaps
            h = min(half, j - start, i - 1 - j)
            out[j] = statistics.median(values[j - h:j + h + 1])
    return out


def weight_summary(frames: list[dict]) -> dict:
    """Shares of frames at weight 0, in between, and at 1, plus how many reach
    the event threshold and the overall coverage."""
    weights = [frame_weight(f) for f in frames]
    n = len(weights) or 1
    return {
        "zero": sum(w == 0 for w in weights) / n,
        "between": sum(0 < w < 1 for w in weights) / n,
        "one": sum(w == 1 for w in weights) / n,
        "event_eligible": sum(w >= config.MIN_EVENT_WEIGHT for w in weights) / n,
        "coverage": _coverage(weights),
    }


def body_confidence_spread(frames: list[dict]) -> dict | None:
    """10th percentile, median and 90th percentile of body confidence on clean
    frames (found, overlap <= WEIGHT_OVERLAP_FULL), to check the ramp limits.
    None if there are fewer than 2 such frames."""
    confs = [body_confidence(f["keypoints"]) for f in frames
             if f["found"] and f["overlap"] <= config.WEIGHT_OVERLAP_FULL]
    if len(confs) < 2:
        return None
    deciles = statistics.quantiles(confs, n=10)
    return {"frames": len(confs), "p10": deciles[0], "median": statistics.median(confs), "p90": deciles[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json", help="Keypoints JSON written by pipeline.analyze")
    parser.add_argument("--frame", type=int, nargs="*", default=[], help="Frames to show the weight breakdown for")
    args = parser.parse_args()

    frames = json.loads(Path(args.json).read_text(encoding="utf-8"))["frames"]
    s = weight_summary(frames)
    print(f"{len(frames)} frames: weight 0 {s['zero']:.1%}, between 0 and 1 {s['between']:.1%}, "
          f"weight 1 {s['one']:.1%}")
    print(f"weight >= {config.MIN_EVENT_WEIGHT} (counted for events): {s['event_eligible']:.1%}; "
          f"coverage {s['coverage']:.2f}")
    spread = body_confidence_spread(frames)
    if spread:
        print(f"body confidence on {spread['frames']} clean frames (overlap <= {config.WEIGHT_OVERLAP_FULL}): "
              f"p10 {spread['p10']:.2f}, median {spread['median']:.2f}, p90 {spread['p90']:.2f}  "
              f"(ramp {config.WEIGHT_CONF_ZERO} -> {config.WEIGHT_CONF_FULL})")
    for n in args.frame:
        f = frames[n]
        if not f["found"]:
            kind = "clinch" if f.get("clinch") else "not found"
            print(f"frame {n}: no skeleton ({kind}) -> weight 0")
            continue
        conf = body_confidence(f["keypoints"])
        print(f"frame {n}: overlap {f['overlap']:.4f} -> factor {overlap_factor(f['overlap']):.2f}; "
              f"body confidence {conf:.2f} -> factor {confidence_factor(conf):.2f}; "
              f"weight {frame_weight(f):.2f}")


if __name__ == "__main__":
    main()
