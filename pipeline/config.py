"""Tunable limits for frame weighting (see pipeline/metrics.py and DECISIONS.md).

A frame's weight says how much a metric should trust it, from 0 (ignore) to 1
(full trust). Changing these takes effect on existing JSON files too: metrics
recompute weights from the stored overlap and keypoint confidences.
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
