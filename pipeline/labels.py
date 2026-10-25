"""Check the metrics against the hand labels in labels/phase2/ (Phase 2).

`python -m pipeline.labels guard` picks the fixed guard threshold (DECISIONS.md
011): the guard height that best separates the hands labelled `up` from the
ones labelled `down`. It prints every labelled hand next to its value, the
results per clip, hand and fighter, and a cross-clip check (fit on one clip,
test on the other).

Labels are copied from outputs/labelling/ (made by pipeline/labelling_kit.py)
so they're kept in git.
"""
import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pipeline.body import HANDS, guard_height
from pipeline.labelling_kit import CLIPS
from pipeline.metrics import median_filter

LABELS = Path("labels/phase2")
GUARD_ANSWERS = ("up", "down")  # `unsure` and `punch` say nothing about the guard


# --- pure, tested -------------------------------------------------------------

@dataclass
class GuardLabel:
    """One hand in one labelled guard frame. Left/right are the fighter's own."""
    id: str
    clip: str
    frame: int
    fighter: str
    hand: str
    label: str  # "up" or "down"


def load_guard_labels(path: Path) -> tuple[list[GuardLabel], Counter]:
    """The `up`/`down` hands from guard_labels.csv, plus how often each answer
    was given (including the `unsure` and `punch` ones left out)."""
    labels, answers = [], Counter()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for hand in HANDS:
                answer = row[f"{hand}_hand"].strip().lower()
                answers[answer] += 1
                if answer in GUARD_ANSWERS:
                    labels.append(GuardLabel(row["id"], row["clip"], int(row["frame"]), row["fighter"], hand, answer))
    return labels, answers


@dataclass
class Split:
    """How well a threshold separates the hands: below it counts as down."""
    threshold: float
    downs_caught: int
    downs: int
    ups_caught: int
    ups: int

    @property
    def balanced_accuracy(self) -> float:
        """Average of the share of downs caught and the share of ups caught,
        so the bigger group doesn't win just by being bigger."""
        shares = [caught / n for caught, n in ((self.downs_caught, self.downs), (self.ups_caught, self.ups)) if n]
        return sum(shares) / len(shares) if shares else 0.0


def evaluate(threshold: float, values: list[float], is_down: list[bool]) -> Split:
    downs = sum(is_down)
    downs_caught = sum(v < threshold for v, d in zip(values, is_down) if d)
    ups_caught = sum(v >= threshold for v, d in zip(values, is_down) if not d)
    return Split(threshold, downs_caught, downs, ups_caught, len(values) - downs)


def pick_threshold(values: list[float], is_down: list[bool]) -> Split:
    """Try every midpoint between neighbouring values and keep the one with
    the best balanced accuracy. On a tie, the one in the widest gap between
    values wins: the threshold sits where there's most room either side."""
    points = sorted(set(values))
    if len(points) < 2 or all(is_down) or not any(is_down):
        raise ValueError("need both up and down hands with different values")
    candidates = [((a + b) / 2, b - a) for a, b in zip(points, points[1:])]
    scored = [(evaluate(t, values, is_down), gap) for t, gap in candidates]
    best, _ = max(scored, key=lambda s: (round(s[0].balanced_accuracy, 9), s[1]))
    return best


# --- reading the JSONs and printing -------------------------------------------

def _short(clip: str) -> str:
    return clip.split()[-1]


def smoothed_guard(json_path: str) -> dict[str, list[float | None]]:
    """Guard height per hand for every frame, after the median filter (the
    value the metric will see)."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    aspect = data["width"] / data["height"]
    return {hand: median_filter([guard_height(f, aspect, hand) for f in data["frames"]], data["fps"])
            for hand in HANDS}


def _line(name: str, s: Split) -> str:
    return (f"  {name:<22} downs {s.downs_caught}/{s.downs}  ups {s.ups_caught}/{s.ups}"
            f"  balanced accuracy {s.balanced_accuracy:.2f}")


def guard_report() -> None:
    labels, answers = load_guard_labels(LABELS / "guard_labels.csv")
    print("Answers per hand:", ", ".join(f"{a} {n}" for a, n in answers.most_common()))
    series = {(clip, name): smoothed_guard(path) for clip, fighters in CLIPS.items() for name, path in fighters.items()}

    rows, no_value = [], []
    for lab in labels:
        value = series[(lab.clip, lab.fighter)][lab.hand][lab.frame]
        (no_value if value is None else rows).append((lab, value))
    if no_value:
        print("No guard height (wrist or shoulder not seen):", ", ".join(f"{l.id} {l.hand}" for l, _ in no_value))

    best = pick_threshold([v for _, v in rows], [l.label == "down" for l, _ in rows])
    t = best.threshold
    print(f"\nThreshold: {t:+.3f} torso lengths (below = down)")
    print(_line("all", best))

    def subset(name, keep):
        part = [(l, v) for l, v in rows if keep(l)]
        print(_line(name, evaluate(t, [v for _, v in part], [l.label == "down" for l, _ in part])))

    print("\nSame threshold, per clip / hand / fighter:")
    for clip in CLIPS:
        subset(_short(clip), lambda l, c=clip: l.clip == c)
    for hand in HANDS:
        subset(f"{hand} hand", lambda l, h=hand: l.hand == h)
    for clip, fighters in CLIPS.items():
        for name in fighters:
            subset(f"{_short(clip)} {name}", lambda l, c=clip, n=name: (l.clip, l.fighter) == (c, n))

    print("\nCross-clip (fit on one clip, test on the other):")
    for fit in CLIPS:
        for test in CLIPS:
            if test == fit:
                continue
            train = [(l, v) for l, v in rows if l.clip == fit]
            held = [(l, v) for l, v in rows if l.clip == test]
            s = pick_threshold([v for _, v in train], [l.label == "down" for l, _ in train])
            on_test = evaluate(s.threshold, [v for _, v in held], [l.label == "down" for l, _ in held])
            print(f"  fit {_short(fit)} -> {s.threshold:+.3f} (fit {s.balanced_accuracy:.2f}),"
                  f" on {_short(test)}: {on_test.balanced_accuracy:.2f}"
                  f" (downs {on_test.downs_caught}/{on_test.downs}, ups {on_test.ups_caught}/{on_test.ups})")

    print("\nAll labelled hands, lowest first (* = on the wrong side of the threshold):")
    for lab, value in sorted(rows, key=lambda r: r[1]):
        wrong = (value < t) != (lab.label == "down")
        print(f"  {'*' if wrong else ' '} {value:+.3f}  {lab.label:<4}  {lab.id}  {_short(lab.clip):<11}"
              f" {lab.fighter}  {lab.hand}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["guard"])
    parser.parse_args()
    guard_report()


if __name__ == "__main__":
    main()
