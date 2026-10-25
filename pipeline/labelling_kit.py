"""Build a hand-labelling kit in outputs/labelling/ (Phase 2, DECISIONS.md 011-012).

1. Guard frames: ~20 JPGs per clip of fighters to judge "hand up / down",
   picked with the guard-height signal (clear up, clear down, ~1/3
   borderline), spread over the clip, both fighters, both hands. Fighters are
   boxed and named; no skeleton or numbers, so the judgement isn't biased.
   guard_labels.csv is prefilled for you to fill in.
2. Punch review videos per clip (normal and 0.5x speed) with time and frame
   number burned in, plus an empty punch_labels.csv.
3. index.html to click through it all, and a README.md.

Both fighters come from their own keypoints JSON, each tracked with kit
identification by pipeline.analyze (run once per fighter, with --opponent set
to the other one). Which frames are repeats of the one before is found by
decoding the clip once (cached in outputs/labelling/cache/).

    python -m pipeline.labelling_kit

Stops if guard_labels.csv already has labels; never overwrites punch_labels.csv.
"""
import argparse
import csv
import html
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline import config
from pipeline.body import HANDS, guard_height

# Per clip, the name shown for each fighter and their keypoints JSON.
# clinch_clip has only fighter_1: tracking the opponent (fighter_4.json) often
# follows another boxer in red headgear in the background instead, as the kit
# check can't tell those two apart (DECISIONS.md 007, "same kit").
CLIPS = {
    "Recording 2026-09-24 104715": {
        "fighter_2": "outputs/phase1_check/Recording 2026-09-24 104715/fighter_2.json",
        "fighter_1": "outputs/phase1_check/Recording 2026-09-24 104715/fighter_1.json",
    },
    "clinch_clip": {
        "fighter_1": "outputs/phase1_check/clinch_clip/fighter_1.json",
    },
}
OUT = Path("outputs/labelling")
FRAMES_PER_CLIP = 20
MIN_GAP_S = 1.5  # between two picked frames of the same clip
REPEAT_DIFF = 0.5  # mean grey-level change below this = same picture as the frame before
VIDEO_WIDTH = 1280

GUARD_COLUMNS = ["id", "clip", "time_s", "frame", "fighter", "left_hand", "right_hand"]
PUNCH_COLUMNS = ["clip", "time_s", "fighter", "hand", "view", "notes"]


# --- frame selection (pure, tested) --------------------------------------

@dataclass
class Candidate:
    """One fighter in one frame, as a possible guard frame."""
    frame: int
    time: float
    fighter: str
    guard: dict[str, float | None]  # per hand, torso lengths above the shoulder
    weight: float
    repeat: bool = False


@dataclass
class Pick:
    candidate: Candidate
    hand: str  # the hand whose guard height put it in its category
    category: str  # "up", "down" or "borderline"


def _quantile(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def select_guard_frames(candidates: list[Candidate], n: int = FRAMES_PER_CLIP,
                        min_gap: float = MIN_GAP_S) -> list[Pick]:
    """Pick `n` guard frames from one clip, sorted by time.

    Categories come from this clip's guard heights over all usable hands:
    lowest 15% = "down", highest 15% = "up", middle 40-60% = "borderline"
    (about a third of the picks). The clip is split into `n` equal time bins
    with one pick each; categories, fighters and hands take turns across the
    bins. A pick prefers its own bin and the planned fighter/hand, then
    relaxes. Skipped: frames under config.MIN_EVENT_WEIGHT, repeated frames,
    and frames within `min_gap` seconds of an earlier pick.
    """
    usable = [c for c in candidates if c.weight >= config.MIN_EVENT_WEIGHT and not c.repeat]
    items = [(c, hand, c.guard[hand]) for c in usable for hand in HANDS if c.guard.get(hand) is not None]
    if not items:
        return []
    values = sorted(g for _, _, g in items)
    down_max, up_min = _quantile(values, 0.15), _quantile(values, 0.85)
    mid_lo, mid_hi, median = _quantile(values, 0.40), _quantile(values, 0.60), statistics.median(values)

    def category(g: float) -> str | None:
        if g <= down_max:
            return "down"
        if g >= up_min:
            return "up"
        if mid_lo <= g <= mid_hi:
            return "borderline"
        return None

    # Plan: ~1/3 borderline, the rest split between up and down, interleaved.
    n_border = round(n / 3)
    counts = {"up": (n - n_border + 1) // 2, "down": (n - n_border) // 2, "borderline": n_border}
    plan: list[str] = []
    while len(plan) < n:
        for cat in ("up", "down", "borderline"):
            if len(plan) < n and plan.count(cat) < counts[cat]:
                plan.append(cat)
    fighters = sorted({c.fighter for c in usable})

    t0, t1 = min(c.time for c in usable), max(c.time for c in usable)
    span = max(t1 - t0, 1e-9)

    def bin_of(c: Candidate) -> int:
        return min(n - 1, int((c.time - t0) / span * n))

    picks: list[Pick] = []
    for slot, cat in enumerate(plan):
        fighter = fighters[slot % len(fighters)]
        hand = HANDS[(slot // 2) % 2]
        pool = [(c, h, g) for c, h, g in items if category(g) == cat
                and all(abs(c.time - p.candidate.time) >= min_gap for p in picks)]
        if not pool:
            continue
        # Closest bin first; within it the planned fighter and hand, then the
        # most typical value for the category.
        def score(item):
            c, h, g = item
            typical = -g if cat == "up" else g if cat == "down" else abs(g - median)
            return abs(bin_of(c) - slot), c.fighter != fighter, h != hand, typical

        c, h, _ = min(pool, key=score)
        picks.append(Pick(c, h, cat))
    return sorted(picks, key=lambda p: p.candidate.time)


# --- clip data -----------------------------------------------------------------

def repeated_frames(video: Path, cache: Path) -> list[bool]:
    """Per frame: is it the same picture as the frame before? Cached."""
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    out: list[bool] = []
    cap = cv2.VideoCapture(str(video))
    prev = None
    while True:
        ok, image = cap.read()
        if not ok:
            break
        grey = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.int16)
        out.append(prev is not None and float(np.abs(grey - prev).mean()) < REPEAT_DIFF)
        prev = grey
    cap.release()
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def candidates_for(fighters: dict[str, dict], repeats: list[bool]) -> list[Candidate]:
    """Every frame of every fighter's JSON as a possible guard frame."""
    out = []
    for name, data in fighters.items():
        aspect = data["width"] / data["height"]
        for f in data["frames"]:
            repeat = f["frame"] < len(repeats) and repeats[f["frame"]]
            out.append(Candidate(f["frame"], f["time"], name, {h: guard_height(f, aspect, h) for h in HANDS},
                                 f["frame_weight"], repeat))
    return out


def _boxes_at(fighters: dict[str, dict], frame: int) -> dict[str, list[float] | None]:
    return {name: data["frames"][frame].get("bbox") if frame < len(data["frames"]) else None
            for name, data in fighters.items()}


# --- drawing ----------------------------------------------------------------

def _text(image, text: str, org: tuple[int, int], scale: float, color=(255, 255, 255)) -> None:
    thickness = max(1, round(scale * 2))
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness * 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _boxes(image, boxes: dict[str, list[float] | None], judge: str | None = None) -> None:
    """Box and name each fighter; the one to judge in thick yellow."""
    h, w = image.shape[:2]
    scale = h / 1000
    for name, box in boxes.items():
        if box is None:
            continue
        x1, y1, x2, y2 = (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
        color = (0, 255, 255) if name == judge else (255, 200, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, max(2, round(scale * (6 if name == judge else 2))))
        _text(image, name, (x1 + 4, max(int(30 * scale), y1 - int(10 * scale))), 0.9 * scale, color)


def _short(clip: str) -> str:
    return clip.split()[-1]


# --- outputs ------------------------------------------------------------------

def _has_labels(path: Path, columns: list[str]) -> bool:
    if not path.exists():
        return False
    with open(path, newline="", encoding="utf-8") as f:
        return any(any((row.get(c) or "").strip() for c in columns) for row in csv.DictReader(f))


def write_guard_frames(clips: dict[str, tuple[Path, dict[str, dict], list[Pick]]]) -> list[dict]:
    rows, n = [], 0
    for clip, (video, fighters, picks) in clips.items():
        cap = cv2.VideoCapture(str(video))
        for pick in picks:
            n += 1
            c = pick.candidate
            gid = f"G{n:03d}"
            cap.set(cv2.CAP_PROP_POS_FRAMES, c.frame)
            ok, image = cap.read()
            if not ok:
                continue
            _boxes(image, _boxes_at(fighters, c.frame), judge=c.fighter)
            s = image.shape[0] / 1000
            _text(image, f"{gid}  {_short(clip)}  {c.time:.2f} s  frame {c.frame}", (int(20 * s), int(60 * s)), 1.6 * s)
            _text(image, f"judge: {c.fighter}", (int(20 * s), int(120 * s)), 1.4 * s, (0, 255, 255))
            name = f"{gid}_{_short(clip)}_{c.time:.2f}s.jpg"
            cv2.imwrite(str(OUT / "guard_frames" / name), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rows.append({"id": gid, "clip": clip, "time_s": f"{c.time:.2f}", "frame": c.frame, "fighter": c.fighter,
                         "file": name, "hand": pick.hand, "category": pick.category,
                         "guard_left": c.guard["left"], "guard_right": c.guard["right"]})
        cap.release()
    return rows


def write_videos(clip: str, video: Path, fighters: dict[str, dict]) -> list[Path]:
    data = next(iter(fighters.values()))
    paths = [OUT / "videos" / f"{_short(clip)}_review.mp4", OUT / "videos" / f"{_short(clip)}_review_half_speed.mp4"]
    if all(p.exists() for p in paths):
        return paths
    cap = cv2.VideoCapture(str(video))
    fps = data["fps"]
    size = (VIDEO_WIDTH, round(data["height"] * VIDEO_WIDTH / data["width"] / 2) * 2)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = [cv2.VideoWriter(str(paths[0]), fourcc, fps, size), cv2.VideoWriter(str(paths[1]), fourcc, fps / 2, size)]
    for f in data["frames"]:
        ok, image = cap.read()
        if not ok:
            break
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        _boxes(image, _boxes_at(fighters, f["frame"]))
        k = size[1] / 1000
        _text(image, f"{_short(clip)}   {f['time']:.2f} s   frame {f['frame']}", (int(20 * k), int(70 * k)), 1.8 * k)
        for w in writers:
            w.write(image)
    for w in writers:
        w.release()
    cap.release()
    return paths


def write_csvs(guard_rows: list[dict]) -> None:
    guard_csv, punch_csv = OUT / "guard_labels.csv", OUT / "punch_labels.csv"
    with open(guard_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, GUARD_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows({**r, "left_hand": "", "right_hand": ""} for r in guard_rows)
    if not punch_csv.exists():
        with open(punch_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(PUNCH_COLUMNS)
    # How each frame was picked, for reading the labels back later. Kept apart
    # from guard_labels.csv so it doesn't bias the labelling.
    with open(OUT / "cache" / "guard_selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["id", "clip", "frame", "fighter", "hand", "category", "guard_left", "guard_right"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(guard_rows)


def write_index(guard_rows: list[dict], videos: list[Path]) -> None:
    parts = ["<!doctype html><meta charset='utf-8'><title>Labelling kit</title>",
             "<style>body{font-family:sans-serif;margin:16px;background:#fafafa;color:#222}"
             ".grid{display:flex;flex-wrap:wrap;gap:12px}.t{width:240px;font-size:13px}"
             ".t img{width:240px;border:1px solid #ccc}a{color:#0645ad}</style>",
             "<h1>Labelling kit</h1><p>Instructions: <a href='README.md'>README.md</a>. "
             "Fill in <code>guard_labels.csv</code> and <code>punch_labels.csv</code>.</p>",
             "<h2>Punch review videos</h2><ul>"]
    for v in videos:
        rel = v.relative_to(OUT).as_posix()
        parts.append(f"<li><a href='{html.escape(rel)}'>{html.escape(v.name)}</a></li>")
    parts.append("</ul><p>If the browser downloads a video instead of playing it, open it in VLC "
                 "(E steps one frame) or Windows Media Player.</p>")
    for clip in CLIPS:
        parts.append(f"<h2>Guard frames: {html.escape(clip)}</h2><div class='grid'>")
        for r in (r for r in guard_rows if r["clip"] == clip):
            src = html.escape(f"guard_frames/{r['file']}")
            parts.append(f"<div class='t'><a href='{src}'><img src='{src}' loading='lazy'></a>"
                         f"<br><b>{r['id']}</b> {r['time_s']} s, frame {r['frame']}, judge {r['fighter']}</div>")
        parts.append("</div>")
    (OUT / "index.html").write_text("\n".join(parts), encoding="utf-8")


README = """# Labelling kit

Open `index.html` to click through everything. Two CSVs to fill in; when
they're done, hand them back and they'll be used to set the guard exposure
threshold (DECISIONS.md 011) and to compare the punch signals (012).

## Which fighter is which

The fighter to judge has a **thick yellow box**; the other one a thin blue one.

- **104715**: fighter_2 is the one tracked in Phase 1 (red headgear, white
  T-shirt); fighter_1 is the opponent (green headgear).
- **clinch_clip**: only fighter_1 (white shirt, white headgear) is boxed and
  judged. The opponent is fighter_2 (red headgear, grey shirt, black shorts);
  there's no box for them because tracking kept mixing them up with another
  boxer in red headgear in the background. When you label punches, that
  background boxer isn't fighter_2.

Both fighters were tracked by their kit (headgear, shirt, trunks), the same
way as in Phase 1. During clinches neither fighter is assigned, so there are
no boxes then; the video itself is unaffected.

## guard_labels.csv: one row per image

Look at the image named in `id` and judge the fighter in the `fighter`
column. For **each hand**, write in `left_hand` / `right_hand`:

- `up`: glove high enough to protect the chin/face
- `down`: glove clearly lower, the chin/face is open on that side
- `unsure`: can't tell (hidden, blurry, mid-movement)

Left and right are the **fighter's own** left and right, not the screen's.
Judge what you see; you haven't been told what the model measured, on
purpose. About a third of the frames were chosen to be borderline, so
`unsure` is a fine answer when it really is in between. Don't open
`cache/guard_selection.csv` until you're done: it says how each frame was
picked.

## punch_labels.csv: one row per punch

Watch `videos/*_review.mp4` (or `_half_speed`, easier for fast
combinations). The time and frame number are burned into every frame. For
each punch, add a row:

- `clip`: 104715 or clinch_clip
- `time_s`: the moment of **full extension**, from the burned-in time
- `fighter`: fighter_1 or fighter_2
- `hand`: left or right (the fighter's own)
- `view`: `side` (punch goes across the screen), `front` (towards or away
  from the camera) or `unsure`
- `notes`: anything useful, e.g. "jab", "hook", "feint", "blocked"

You don't have to cover the whole video: a few stretches of 20-30 s with
every punch in them are more useful than scattered punches. Write down which
stretches you covered in `notes` of the first row.

Videos are MP4 (MPEG-4). If the browser downloads instead of playing, use
VLC (E steps one frame forward) or Windows Media Player.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    if _has_labels(OUT / "guard_labels.csv", ["left_hand", "right_hand"]):
        # Rebuilding could renumber the images, so the labels would point at the wrong frames.
        raise SystemExit(f"{OUT / 'guard_labels.csv'} already has labels; move it elsewhere to rebuild the kit")
    for sub in ("guard_frames", "videos", "cache"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    clips, videos = {}, []
    for clip, paths in CLIPS.items():
        print(f"== {clip}")
        fighters = {name: json.loads(Path(p).read_text(encoding="utf-8")) for name, p in paths.items()}
        video = Path("data") / f"{clip}.mp4"
        repeats = repeated_frames(video, OUT / "cache" / f"{_short(clip)}_repeats.json")
        picks = select_guard_frames(candidates_for(fighters, repeats))
        split = {cat: sum(p.category == cat for p in picks) for cat in ("up", "down", "borderline")}
        print(f"  {len(picks)} guard frames: {split}")
        clips[clip] = (video, fighters, picks)
        videos += write_videos(clip, video, fighters)

    for old in (OUT / "guard_frames").glob("*.jpg"):
        old.unlink()  # only images this script wrote; ids may have moved
    rows = write_guard_frames(clips)
    write_csvs(rows)
    write_index(rows, videos)
    (OUT / "README.md").write_text(README, encoding="utf-8")
    print(f"done -> {OUT.resolve()}")


if __name__ == "__main__":
    main()
