"""Kit-based identity: tell the chosen fighter from the opponent by what they
wear (headgear, shirt, trunks).

Each kit region is cropped using the pose keypoints and summarised as a
colour histogram: hue-saturation bins for coloured pixels plus a few
brightness bins for colourless ones (black / grey / white), so both a red vs
blue headgear and a white vs black shirt count as differences.

At the start, a kit comparison measures how different each region is between
the two fighters and weights the regions by it, so whatever tells them apart
best in *this* video (often the headgear) decides who is who.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from pipeline.common.keypoints import PersonPose

KIT_REGIONS = ("headgear", "shirt", "trunks")
MIN_CONF = 0.3  # keypoint confidence needed to use a keypoint for cropping
MIN_SATURATION = 60  # below this a pixel counts as colourless (white/grey/black)
MIN_VALUE = 40  # below this it is too dark to have a reliable hue
HS_BINS = [16, 8]  # hue, saturation bins for coloured pixels
GREY_BINS = 4  # brightness bins for colourless pixels
MIN_PIXELS = 30  # smaller crops are skipped rather than guessed

Signature = dict[str, np.ndarray]


def _points(frame_shape, person: PersonPose, indices: list[int]) -> list[tuple[float, float]]:
    h, w = frame_shape[:2]
    return [(person.keypoints[i].x * w, person.keypoints[i].y * h)
            for i in indices if i < len(person.keypoints) and person.keypoints[i].confidence >= MIN_CONF]


def region_boxes(frame_shape, person: PersonPose) -> dict[str, tuple[int, int, int, int]]:
    """Pixel boxes (x1, y1, x2, y2) of each visible kit region.

    - headgear: above and around the face points (nose, eyes, ears), so it
      covers the headgear rather than the face;
    - shirt: shoulders to hips; trunks: hips to knees.
    """
    h, w = frame_shape[:2]
    boxes = {}
    face = _points(frame_shape, person, [0, 1, 2, 3, 4])
    if len(face) >= 2 and person.bbox is not None:
        xs, ys = zip(*face)
        person_h = (person.bbox[3] - person.bbox[1]) * h
        # Head size: face points span little in profile, so fall back on body height.
        s = max(max(xs) - min(xs), max(ys) - min(ys), 0.12 * person_h)
        cx, top = float(np.mean(xs)), min(ys)
        boxes["headgear"] = (cx - 0.8 * s, top - 1.1 * s, cx + 0.8 * s, top + 0.3 * s)
    for name, indices in (("shirt", [5, 6, 11, 12]), ("trunks", [11, 12, 13, 14])):
        pts = _points(frame_shape, person, indices)
        if len(pts) >= 3:
            xs, ys = zip(*pts)
            boxes[name] = (min(xs), min(ys), max(xs), max(ys))

    out = {}
    for name, (x1, y1, x2, y2) in boxes.items():
        x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            out[name] = (x1, y1, x2, y2)
    return out


def region_histogram(crop: np.ndarray) -> np.ndarray | None:
    """Normalized histogram: hue-saturation bins for coloured pixels followed by
    brightness bins for colourless ones. None if the crop is too small."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    coloured = cv2.inRange(hsv, (0, MIN_SATURATION, MIN_VALUE), (180, 255, 255))
    hs = cv2.calcHist([hsv], [0, 1], coloured, HS_BINS, [0, 180, 0, 256]).ravel()
    grey = cv2.calcHist([hsv], [2], cv2.bitwise_not(coloured), [GREY_BINS], [0, 256]).ravel()
    hist = np.concatenate([hs, grey]).astype(np.float32)
    total = hist.sum()
    return hist / total if total >= MIN_PIXELS else None


def kit_signature(frame: np.ndarray, person: PersonPose) -> Signature:
    """Histogram per visible kit region; hidden regions are left out."""
    signature = {}
    for name, (x1, y1, x2, y2) in region_boxes(frame.shape, person).items():
        hist = region_histogram(frame[y1:y2, x1:x2])
        if hist is not None:
            signature[name] = hist
    return signature


def histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance: 0 = same colours, 1 = nothing in common."""
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


@dataclass
class KitProfile:
    """What the fighter and the opponent wear, and how much each kit region
    counts when deciding who is who."""
    fighter: Signature
    opponent: Signature
    weights: dict[str, float]  # region -> weight, summing to 1
    # region -> (fighter-vs-opponent distance, fighter's own spread, separation), None if not seen
    report: dict[str, tuple[float, float, float] | None] = field(default_factory=dict)

    def distances(self, signature: Signature) -> tuple[float, float] | None:
        """Weighted distance of `signature` to (fighter, opponent), over the
        weighted regions it has. None if it has none of them."""
        regions = [r for r in self.weights if r in signature]
        if not regions:
            return None
        total = sum(self.weights[r] for r in regions)
        to_fighter = sum(self.weights[r] * histogram_distance(signature[r], self.fighter[r]) for r in regions)
        to_opponent = sum(self.weights[r] * histogram_distance(signature[r], self.opponent[r]) for r in regions)
        return to_fighter / total, to_opponent / total

    def adapt(self, which: str, signature: Signature, alpha: float = 0.05) -> None:
        """Move the fighter's or opponent's prototype a little towards
        `signature` (lighting changes, scene cuts)."""
        proto = self.fighter if which == "fighter" else self.opponent
        for region, hist in signature.items():
            if region in proto:
                proto[region] = (1 - alpha) * proto[region] + alpha * hist

    def describe(self) -> str:
        parts = [f"{r} {'n/a' if self.report.get(r) is None else f'{self.report[r][2]:+.2f}'}"
                 for r in KIT_REGIONS]
        using = ", ".join(f"{r} ({w:.0%})" for r, w in sorted(self.weights.items(), key=lambda rw: -rw[1]))
        return f"kit comparison: {', '.join(parts)} -> using {using or 'nothing'}"


def build_kit_profile(fighter_sigs: list[Signature], opponent_sigs: list[Signature],
                      min_separation: float = 0.10, min_samples: int = 5) -> KitProfile | None:
    """Compare the two fighters' kit over several frames.

    Per region, separation = distance between the fighters' average kit minus
    the fighter's own frame-to-frame spread. Regions that separate them by less
    than `min_separation` are dropped; the rest are weighted by separation.
    Returns None if no region separates them.
    """
    fighter, opponent, weights, report = {}, {}, {}, {}
    for region in KIT_REGIONS:
        f = [s[region] for s in fighter_sigs if region in s]
        o = [s[region] for s in opponent_sigs if region in s]
        if len(f) < min_samples or len(o) < min_samples:
            report[region] = None
            continue
        fighter[region], opponent[region] = np.mean(f, axis=0), np.mean(o, axis=0)
        between = histogram_distance(fighter[region], opponent[region])
        spread = float(np.mean([histogram_distance(s, fighter[region]) for s in f]))
        report[region] = (between, spread, between - spread)
        if between - spread >= min_separation:
            weights[region] = between - spread
    if not weights:
        return None
    total = sum(weights.values())
    weights = {r: w / total for r, w in weights.items()}
    return KitProfile(fighter=fighter, opponent=opponent, weights=weights, report=report)
