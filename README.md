# Boxing Coach AI

Analyzes sparring footage of one selected fighter and gives offense/defense
feedback, with progress tracking across sessions. Currently in Phase 1: a
command-line pipeline that tracks the chosen fighter and draws their skeleton.
See [SPEC.md](SPEC.md) for the phases and [DECISIONS.md](DECISIONS.md) for why
things are built the way they are.

## Setup (Windows, Python 3.11)

```
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
```

Torch is installed on its own because it needs a CUDA-specific build. The
`cu128` build is required for RTX 50-series (Blackwell) GPUs; see
DECISIONS.md 005.

## Usage

1. Pick a fighter. This saves `outputs/analyze/<clip>/preview.jpg` with each
   person numbered:

   ```
   .venv\Scripts\python -m pipeline.analyze "data/clip.mp4" --preview
   ```

2. Draw that fighter's skeleton on the video (use the same `--start` as the
   preview, if you set one):

   ```
   .venv\Scripts\python -m pipeline.analyze "data/clip.mp4" --fighter 1
   ```

## What you get

Each run writes to `outputs/analyze/<clip>/`:

- **`fighter_N.mp4`**: the video with the fighter's skeleton drawn in cyan.
  No skeleton means the fighter is out of view, or they're in a clinch. In a
  clinch the two fighters can't be told apart reliably, so nothing is drawn
  until they separate (DECISIONS.md 009).
- **`fighter_N.json`**: the same thing as numbers, one entry per frame:
  - `found`: whether the fighter's skeleton is there
  - `keypoints`: 17 body points as `[x, y, confidence]`, with x and y as
    fractions of the frame (0–1)
  - `overlap`: how much the fighter's box overlaps the opponent's (0 = apart)
  - `clinch`: true during a clinch
  - `frame_weight`: how much a metric should trust this frame, from 0 to 1
    (DECISIONS.md 010)

  The top of the file has the video's fps and size, the names of the 17
  points, and a list of every clinch with how it ended.

To see how trustworthy a clip's frames are, and why a given frame got its
weight:

```
.venv\Scripts\python -m pipeline.metrics "outputs/analyze/clip/fighter_1.json" --frame 120
```

## Tests

```
.venv\Scripts\python -m pytest tests
```

## Not in this repo

- **Videos** (`data/`, `outputs/`): large, and some footage isn't mine to
  publish (DECISIONS.md 004).
- **Model weights** (`*.pt`): ultralytics downloads `yolov8s-pose.pt`
  automatically on first run.
