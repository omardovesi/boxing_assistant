"""Tunable limits for metrics (see pipeline/metrics.py, pipeline/body.py and
DECISIONS.md). Metrics read the keypoints JSON, so changing these takes effect
on existing files without re-running the video.

A frame's weight says how much a metric should trust it, from 0 (ignore) to 1
(full trust), recomputed from the stored overlap and keypoint confidences.
"""

# Overlap with the opponent's box (IoU): full weight up to FULL, falling
# linearly to zero weight at ZERO, where the fighters are tangled up.
WEIGHT_OVERLAP_FULL = 0.10
WEIGHT_OVERLAP_ZERO = 0.30

# Mean confidence of the 12 body keypoints: zero weight up to ZERO, rising
# linearly to full weight at FULL. FULL is about the 10th percentile of clean
# (non-overlapping) frames on clinch_clip (0.54).
WEIGHT_CONF_ZERO = 0.30
WEIGHT_CONF_FULL = 0.55

# Event counts (e.g. punches) ignore frames lighter than this.
MIN_EVENT_WEIGHT = 0.5

# Keypoints less confident than this are treated as missing by the body
# measurements (pipeline/body.py). Same as the drawing threshold.
MIN_KEYPOINT_CONF = 0.3

# Fixed guard standard, in torso lengths above the shoulder line: a hand below
# it counts as down for the "exposure" number. Picked from 63 hand-labelled
# hands in labels/phase2/ (python -m pipeline.labels guard, DECISIONS.md 011).
GUARD_EXPOSURE_THRESHOLD = -0.34

# Punch signals (pipeline/signals.py, DECISIONS.md 012): wrist speed is the
# distance moved over SPEED_SPAN, elbow opening the angle gained over
# OPENING_SPAN. Spans in seconds, not frames, because 12-33% of frames are
# repeats of the one before.
SPEED_SPAN_SECONDS = 0.1
OPENING_SPAN_SECONDS = 0.2

# Median filter length, in seconds (0.1 s = 3 frames at 30 fps). Removes
# one-frame keypoint jumps (e.g. a wrist landing below its own elbow) while a
# short jab survives. In seconds so it means the same at any frame rate.
SMOOTH_SECONDS = 0.1
