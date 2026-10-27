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

from pipeline import config
from pipeline.body import HANDS, guard_height
from pipeline.labelling_kit import CLIPS
from pipeline.metrics import frame_weight, median_filter
from pipeline.signals import SIGNALS

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


def parse_time(text: str) -> float:
    """'1:35.36' -> 95.36 s, '00:21' -> 21.0 s, '12.5' -> 12.5 s."""
    seconds = 0.0
    for part in text.strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


# Who the punch labels name by headgear colour, per clip (labelling README).
# None = labelled, but that fighter has no keypoints JSON.
FIGHTER_BY_KIT = {
    "Recording 2026-09-24 104715": {"red headgear": "fighter_2", "green headgear": "fighter_1"},
    "clinch_clip": {"white headgear": "fighter_1", "red headgear": None},
}
NUMBERS = ("first", "second", "third", "fourth", "fifth")


def punch_kind(notes: str) -> str | None:
    """Boxing number (1 jab, 2 cross, 3 lead hook, 4 rear hook...) from notes
    like '2 to the body' or 'second of 1-2' (-> the 2nd punch of 1-2)."""
    words = notes.strip().lower().split()
    if not words:
        return None
    if words[0].isdigit():
        return words[0]
    if words[0] in NUMBERS and len(words) >= 3 and words[1] == "of":
        combo = words[2].split("-")
        n = NUMBERS.index(words[0])
        return combo[n] if n < len(combo) and combo[n].isdigit() else None
    return None


@dataclass
class PunchLabel:
    clip: str
    raw_time: str
    time: float
    exact: bool  # written with decimals; whole seconds are only good to ~1 s
    fighter: str
    hand: str
    view: str
    kind: str | None


def load_punch_labels(path: Path) -> tuple[list[PunchLabel], list[dict]]:
    """Punches whose fighter has a keypoints JSON, plus the rows skipped."""
    labels, skipped = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fighter = FIGHTER_BY_KIT.get(row["clip"], {}).get(row["fighter"].strip().lower())
            if fighter is None:
                skipped.append(row)
                continue
            raw = row["time_s"].strip()
            labels.append(PunchLabel(row["clip"], raw, parse_time(raw), "." in raw, fighter,
                                     row["hand"].strip().lower().removesuffix("_hand"),
                                     row["view"].strip().lower(), punch_kind(row["notes"])))
    return labels, skipped


WINDOW = 1.6  # seconds; every window (punch or not) is this long, so none gets more chances to score high


def punch_window(label: PunchLabel) -> tuple[float, float]:
    """WINDOW seconds around the labelled moment to look for the punch. A
    whole-second time may have been rounded down from anywhere in that second,
    so its window starts 0.3 s before and runs past the end of that second."""
    start = label.time - 0.3 if not label.exact else label.time - WINDOW / 2
    return start, start + WINDOW


def auc(positives: list[float], negatives: list[float]) -> float:
    """Chance that a random punch window scores higher than a random no-punch
    window (ties count half): 0.5 = no better than guessing, 1 = perfect."""
    if not positives or not negatives:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


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


CLEAR_OF_PUNCHES = 1.0  # seconds a no-punch window keeps from any labelled punch by that fighter


def shared_frames(fighters: dict[str, dict]) -> set[int]:
    """Frames where two fighters' JSONs follow the same person (the kit check
    put both on one track, e.g. 104715 frames 3835-3839): trust neither."""
    tracks = [{f["frame"]: f.get("track_id") for f in d["frames"] if f["found"]} for d in fighters.values()]
    if len(tracks) < 2:
        return set()
    return {fr for fr, t in tracks[0].items() if tracks[1].get(fr) == t}


def _window_score(series: list[float | None], usable: list[bool], fps: float, t0: float, t1: float) -> float | None:
    """Highest value of a signal on trusted frames between t0 and t1 seconds."""
    vals = [series[i] for i in range(max(0, round(t0 * fps)), min(len(series), round(t1 * fps) + 1))
            if usable[i] and series[i] is not None]
    return max(vals) if vals else None


def _punch_line(name: str, s: Split) -> str:
    return (f"  {name:<24} punches caught {s.downs_caught}/{s.downs}  no-punch windows quiet {s.ups_caught}/{s.ups}"
            f"  balanced accuracy {s.balanced_accuracy:.2f}")


def _fit_above(pos: list[float], neg: list[float]) -> Split:
    """Best threshold where *above* = punch (pick_threshold works on below)."""
    s = pick_threshold([-v for v in pos + neg], [True] * len(pos) + [False] * len(neg))
    s.threshold = -s.threshold
    return s


def _score_above(threshold: float, pos: list[float], neg: list[float]) -> Split:
    return Split(threshold, sum(v > threshold for v in pos), len(pos), sum(v <= threshold for v in neg), len(neg))


def punch_report() -> None:
    labels, skipped = load_punch_labels(LABELS / "punch_labels.csv")
    print(f"Punch labels: {len(labels)} used, {len(skipped)} skipped (fighter has no keypoints JSON: "
          f"{', '.join(sorted({r['clip'] + ' ' + r['fighter'] for r in skipped}))})")
    data = {clip: {name: json.loads(Path(p).read_text(encoding="utf-8")) for name, p in fighters.items()}
            for clip, fighters in CLIPS.items()}
    print("\nConverted times (check these):")
    for lab in [l for l in labels if not l.exact][:4] + [l for l in labels if l.exact][:4]:
        fps = next(iter(data[lab.clip].values()))["fps"]
        print(f"  {_short(lab.clip):<11} {lab.raw_time:>8} -> {lab.time:7.2f} s -> frame {round(lab.time * fps):>5}"
              f"  {lab.fighter} {lab.hand} {lab.view} kind {lab.kind}")
    positives, negatives, unmeasured = [], [], []  # (key, {signal: score}), key = (clip, fighter, hand, label)
    for clip, fighters in data.items():
        shared = shared_frames(fighters)
        for name, d in fighters.items():
            frames, fps, aspect = d["frames"], d["fps"], d["width"] / d["height"]
            usable = [frame_weight(f) >= config.MIN_EVENT_WEIGHT and f["frame"] not in shared for f in frames]
            mine = [l for l in labels if (l.clip, l.fighter) == (clip, name)]
            busy = [(a - CLEAR_OF_PUNCHES, b + CLEAR_OF_PUNCHES) for a, b in map(punch_window, mine)]
            for hand in HANDS:
                series = {s: fn(frames, aspect, fps, hand) for s, fn in SIGNALS.items()}
                for lab in (l for l in mine if l.hand == hand):
                    scores = {s: _window_score(v, usable, fps, *punch_window(lab)) for s, v in series.items()}
                    (positives if any(v is not None for v in scores.values()) else unmeasured).append(
                        ((clip, name, hand, lab), scores))
                t, end = 0.0, len(frames) / fps
                while t + WINDOW <= end:
                    t1 = t + WINDOW
                    i0, i1 = round(t * fps), round(t1 * fps)
                    if not any(a < t1 and t < b for a, b in busy) and sum(usable[i0:i1]) >= (i1 - i0) / 2:
                        negatives.append(((clip, name, hand, None),
                                          {s: _window_score(v, usable, fps, t, t1) for s, v in series.items()}))
                    t = t1
    print(f"\n{len(positives)} punches measurable, {len(unmeasured)} not (no trusted frame in their window: "
          f"{', '.join(f'{_short(k[0])} {k[3].raw_time}' for k, _ in unmeasured)}); "
          f"{len(negatives)} no-punch windows of {WINDOW} s")

    for sig in SIGNALS:
        pos = [(k, s[sig]) for k, s in positives if s[sig] is not None]
        neg = [(k, s[sig]) for k, s in negatives if s[sig] is not None]
        best = _fit_above([v for _, v in pos], [v for _, v in neg])
        t = best.threshold
        print(f"\n=== {sig}: AUC {auc([v for _, v in pos], [v for _, v in neg]):.2f},"
              f" best threshold {t:.3f} (above = punch)")
        print(_punch_line("all", best))

        def part(name, keep_pos, keep_neg=None):
            p = [v for k, v in pos if keep_pos(k)]
            n = [v for k, v in neg if (keep_neg or keep_pos)(k)]
            print(_punch_line(name, _score_above(t, p, n)))

        for clip in CLIPS:
            part(_short(clip), lambda k, c=clip: k[0] == c)
        for hand in HANDS:
            part(f"{hand} hand", lambda k, h=hand: k[2] == h)
        for clip, fighters in CLIPS.items():
            for name in fighters:
                part(f"{_short(clip)} {name}", lambda k, c=clip, n=name: k[:2] == (c, n))
        for view in sorted({k[3].view for k, _ in pos}):
            part(f"view {view}", lambda k, v=view: k[3].view == v, lambda k: True)
        for kind in sorted({k[3].kind or "?" for k, _ in pos}):
            part(f"kind {kind}", lambda k, n=kind: (k[3].kind or "?") == n, lambda k: True)
        for fit in CLIPS:
            for test in CLIPS:
                if test != fit:
                    s = _fit_above([v for k, v in pos if k[0] == fit], [v for k, v in neg if k[0] == fit])
                    on = _score_above(s.threshold, [v for k, v in pos if k[0] == test], [v for k, v in neg if k[0] == test])
                    print(f"  cross-clip: fit {_short(fit)} -> {s.threshold:.3f} (fit {s.balanced_accuracy:.2f}),"
                          f" on {_short(test)}: {on.balanced_accuracy:.2f}"
                          f" (caught {on.downs_caught}/{on.downs}, quiet {on.ups_caught}/{on.ups})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["guard", "punch"])
    args = parser.parse_args()
    guard_report() if args.what == "guard" else punch_report()


if __name__ == "__main__":
    main()
