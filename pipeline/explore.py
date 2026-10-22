"""Phase 2, step 1: look at the raw signals before choosing any thresholds.

Plots, per hand, over time:
- guard height: how far the wrist is above the shoulder line, in torso
  lengths (0 = wrist level with the shoulder, negative = below it)
- arm extension: shoulder-to-wrist distance, in torso lengths
and shades frames the metrics shouldn't trust (frame weight 0, e.g. clinches)
in gray. Also prints typical values per hand, to pick thresholds from.

    python -m pipeline.explore "outputs/analyze/clip/fighter_1.json"
    python -m pipeline.explore "outputs/analyze/clip/fighter_1.json" --start 40 --duration 20

Throwaway exploration: the geometry here moves into pipeline/body.py, with
tests, in step 2.
"""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNGs only, no window
import matplotlib.pyplot as plt

from pipeline import config
from pipeline.common.keypoints import COCO17_NAMES

MIN_CONF = 0.3  # keypoints less confident than this are treated as missing
HANDS = ("left", "right")
COLORS = {"left": "tab:blue", "right": "tab:orange"}


def _point(frame: dict, name: str, aspect: float) -> tuple[float, float] | None:
    """(x, y) of a keypoint with x scaled so x and y use the same unit (the
    frame height), or None if missing or not confident."""
    x, y, conf = frame["keypoints"][COCO17_NAMES.index(name)]
    return (x * aspect, y) if conf >= MIN_CONF else None


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _mid(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def hand_signals(frame: dict, aspect: float) -> dict[str, tuple[float | None, float | None]]:
    """Per hand: (guard height, arm extension) in torso lengths, None where
    the needed keypoints are missing."""
    out = {hand: (None, None) for hand in HANDS}
    if not frame["found"]:
        return out
    p = {n: _point(frame, n, aspect) for n in COCO17_NAMES[5:13]}  # shoulders..hips
    if None in (p["left_shoulder"], p["right_shoulder"], p["left_hip"], p["right_hip"]):
        return out
    torso = _dist(_mid(p["left_shoulder"], p["right_shoulder"]), _mid(p["left_hip"], p["right_hip"]))
    if torso <= 0:
        return out
    for hand in HANDS:
        shoulder, wrist = p[f"{hand}_shoulder"], p[f"{hand}_wrist"]
        if wrist is not None:
            # Image y grows downwards, so shoulder_y - wrist_y > 0 means the hand is higher.
            out[hand] = ((shoulder[1] - wrist[1]) / torso, _dist(shoulder, wrist) / torso)
    return out


def _spread(values: list[float]) -> str:
    if len(values) < 10:
        return "too few frames"
    d = statistics.quantiles(values, n=10)
    return f"p10 {d[0]:+.2f}  median {statistics.median(values):+.2f}  p90 {d[-1]:+.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json", help="Keypoints JSON written by pipeline.analyze")
    parser.add_argument("--start", type=float, default=None, help="Window start, in seconds")
    parser.add_argument("--duration", type=float, default=None, help="Window length, in seconds")
    args = parser.parse_args()

    path = Path(args.json)
    data = json.loads(path.read_text(encoding="utf-8"))
    aspect = data["width"] / data["height"]
    frames = data["frames"]
    if args.start is not None or args.duration is not None:
        t0 = data["start"] if args.start is None else args.start
        t1 = float("inf") if args.duration is None else t0 + args.duration
        frames = [f for f in frames if t0 <= f["time"] < t1]
    if not frames:
        raise SystemExit("no frames in that window")

    times = [f["time"] for f in frames]
    signals = [hand_signals(f, aspect) for f in frames]
    weights = [f["frame_weight"] for f in frames]

    # Typical values, from frames a metric would trust for events.
    print(f"{len(frames)} frames, {times[0]:.1f}-{times[-1]:.1f} s")
    usable = [s for s, w in zip(signals, weights) if w >= config.MIN_EVENT_WEIGHT]
    print(f"usable (weight >= {config.MIN_EVENT_WEIGHT}): {len(usable)} frames")
    for hand in HANDS:
        guard = [s[hand][0] for s in usable if s[hand][0] is not None]
        ext = [s[hand][1] for s in usable if s[hand][1] is not None]
        print(f"  {hand:5} guard height    {_spread(guard)}")
        print(f"  {hand:5} arm extension   {_spread(ext)}")

    width = min(60.0, max(12.0, (times[-1] - times[0]) / 4))  # about 4 s per inch, capped
    fig, axes = plt.subplots(3, 1, figsize=(width, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1]})
    for ax, idx, label in ((axes[0], 0, "guard height\n(torso lengths above shoulder)"),
                           (axes[1], 1, "arm extension\n(torso lengths)")):
        for hand in HANDS:
            ax.plot(times, [s[hand][idx] for s in signals], color=COLORS[hand], lw=0.8, label=hand)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        # Gray where the frame weight is 0: nothing here would count.
        ax.fill_between(times, 0, 1, where=[w == 0 for w in weights], color="0.85",
                        transform=ax.get_xaxis_transform(), step="mid", lw=0)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].legend(loc="upper right")
    axes[2].plot(times, weights, color="k", lw=0.8)
    axes[2].set_ylabel("frame weight")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("time (s)")
    axes[2].set_xlim(times[0], times[-1])
    fig.suptitle(f"{path.parent.name} - {path.stem}")
    fig.tight_layout()

    suffix = "" if args.start is None and args.duration is None else f"_{times[0]:.0f}-{times[-1]:.0f}s"
    out = path.with_name(f"{path.stem}_signals{suffix}.png")
    fig.savefig(out, dpi=100)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
