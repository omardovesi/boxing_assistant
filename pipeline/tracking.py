"""Helpers for picking one tracked fighter out of each frame's detections."""
from collections import deque

import numpy as np

from pipeline.appearance import KitProfile, Signature, build_kit_profile, kit_signature
from pipeline.common.keypoints import PersonPose


def select_fighter(people: list[PersonPose], fighter_id: int) -> PersonPose | None:
    """Return the person with `fighter_id` in this frame, or None if the
    tracker didn't see them (e.g. fully occluded during a clinch)."""
    for person in people:
        if person.track_id == fighter_id:
            return person
    return None


def track_ids(people: list[PersonPose]) -> list[int]:
    """Sorted track IDs present in this frame, ignoring untracked people."""
    return sorted(p.track_id for p in people if p.track_id is not None)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes: 0 = apart, 1 = identical."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _height(person: PersonPose) -> float:
    return person.bbox[3] - person.bbox[1]


def nearest_opponent(people: list[PersonPose], fighter: PersonPose) -> PersonPose | None:
    """The other tracked person of comparable size closest to the fighter:
    in sparring footage that's the opponent, not a bystander further back."""
    candidates = [p for p in people if p is not fighter and p.track_id is not None and p.bbox is not None
                  and _height(p) > 0.6 * _height(fighter)]
    fx, fy = (fighter.bbox[0] + fighter.bbox[2]) / 2, (fighter.bbox[1] + fighter.bbox[3]) / 2
    return min(candidates, default=None,
               key=lambda p: abs((p.bbox[0] + p.bbox[2]) / 2 - fx) + abs((p.bbox[1] + p.bbox[3]) / 2 - fy))


class FighterFollower:
    """Follows one fighter across frames, even when the tracker loses them,
    gives them a new ID, or swaps IDs with the opponent in a clinch.

    1. Calibration: follow the chosen track ID while collecting the kit
       (headgear, shirt, trunks) of the fighter and the opponent on frames where
       they don't overlap. After `calibration_frames` such frames, a kit
       comparison decides which regions tell them apart (see appearance.py).
       Until then, a lost fighter is only re-linked to a *new* ID nearby of
       similar size, as there's nothing yet to tell the fighters apart.

    2. Kit identification: every frame, each person of plausible size is
       scored on whether their kit looks more like the fighter's or the
       opponent's. A person counts as the fighter when their kit is closer to
       the fighter's by `margin`, within `max_fighter_dist` of it, and at least
       `min_opponent_height` x the opponent's height (rules out bystanders
       further back). Then:
       - fighter's ID missing: follow whoever counts as the fighter (relink);
       - someone else counts as the fighter and the followed person doesn't,
         for `confirm_frames` frames in a row: follow them (ID swap);
       - the followed person clearly looks like the opponent for
         `confirm_frames` frames and nobody counts as the fighter: report the
         fighter missing rather than draw the skeleton on the wrong person.
       Both kit prototypes adapt slowly (lighting, scene cuts), but only from
       frames where the identification is confident.

    Without frames (colour unavailable) it stays in the calibration behaviour.
    """

    def __init__(
        self,
        fighter_id: int,
        aspect: float = 1.0,
        max_dist: float = 1.0,
        size_ratio: tuple[float, float] = (0.5, 2.0),
        nearby_frames: int = 90,
        clean_iou: float = 0.1,
        height_window: int = 15,
        relink: bool = True,
        opponent_id: int | None = None,
        calibration_frames: int = 30,
        margin: float = 0.10,
        max_fighter_dist: float = 0.45,
        min_opponent_height: float = 0.7,
        confirm_frames: int = 5,
        adapt_alpha: float = 0.05,
    ):
        self.fighter_id = fighter_id
        self.aspect = aspect  # frame width / height, since bboxes are normalized per axis
        self.max_dist = max_dist  # in multiples of the fighter's recent median height
        self.size_ratio = size_ratio
        self.nearby_frames = nearby_frames
        self.clean_iou = clean_iou
        self.relink = relink
        self.opponent_id = opponent_id
        self.calibration_frames = calibration_frames
        self.margin = margin
        self.max_fighter_dist = max_fighter_dist
        self.min_opponent_height = min_opponent_height
        self.confirm_frames = confirm_frames
        self.adapt_alpha = adapt_alpha
        self.kit: KitProfile | None = None
        self.kit_failed = False  # calibration finished but no kit region separates the fighters
        # (frame, old ID, new ID, new person's kit distance to the fighter, or None if not checked)
        self.relinks: list[tuple[int, int, int, float | None]] = []
        # (frame, old ID, new ID, followed person's kit distance or None, new person's kit distance)
        self.swaps: list[tuple[int, int, int, float | None, float]] = []
        self._frame = -1
        self._first_seen: dict[int, int] = {}
        self._last_bbox: tuple[float, float, float, float] | None = None
        self._heights: deque[float] = deque(maxlen=height_window)
        self._lost_at: int | None = None
        self._fighter_samples: list[Signature] = []
        self._opponent_samples: list[Signature] = []
        self._swap_candidate: int | None = None
        self._swap_streak = 0
        self._doubt_streak = 0

    def update(self, people: list[PersonPose], frame: np.ndarray | None = None) -> PersonPose | None:
        """Feed one frame's people (and the frame itself, to enable kit
        identification); return the fighter, or None if not found."""
        self._frame += 1
        for tid in track_ids(people):
            self._first_seen.setdefault(tid, self._frame)

        if self.kit is not None and frame is not None:
            fighter = self._identify(people, frame)
        else:
            fighter = select_fighter(people, self.fighter_id)
            if fighter is None:
                fighter = self._relink_nearby(people)
            elif frame is not None and self.kit is None and not self.kit_failed:
                self._calibrate(fighter, people, frame)
        if fighter is None:
            return None

        self._last_bbox = fighter.bbox
        self._lost_at = None
        if fighter.bbox is not None and self._is_clean(fighter, people):
            self._heights.append(_height(fighter))
        return fighter

    # --- calibration -----------------------------------------------------

    def _calibrate(self, fighter: PersonPose, people: list[PersonPose], frame: np.ndarray) -> None:
        if fighter.bbox is None:
            return
        if self.opponent_id is not None:
            opponent = select_fighter(people, self.opponent_id)
        else:
            opponent = nearest_opponent(people, fighter)
        if opponent is None or opponent.bbox is None or bbox_iou(fighter.bbox, opponent.bbox) >= self.clean_iou:
            return
        self.opponent_id = opponent.track_id
        self._fighter_samples.append(kit_signature(frame, fighter))
        self._opponent_samples.append(kit_signature(frame, opponent))
        if len(self._fighter_samples) >= self.calibration_frames:
            self.kit = build_kit_profile(self._fighter_samples, self._opponent_samples)
            self.kit_failed = self.kit is None
            self._fighter_samples, self._opponent_samples = [], []

    def _relink_nearby(self, people: list[PersonPose]) -> PersonPose | None:
        """Before kit identification: take a *new* ID near where the fighter
        was lost, of similar size; give up after `nearby_frames`."""
        if self._lost_at is None:
            self._lost_at = self._frame
        if (not self.relink or self._last_bbox is None
                or self._frame - self._lost_at > self.nearby_frames):
            return None
        candidates = [
            (self._distance(p.bbox), p)
            for p in people
            if p.track_id is not None
            and p.bbox is not None
            and self._first_seen[p.track_id] >= self._lost_at
            and self._similar_size(p.bbox)
        ]
        candidates = [(d, p) for d, p in candidates if d <= self.max_dist]
        if not candidates:
            return None
        _, best = min(candidates, key=lambda c: c[0])
        self.relinks.append((self._frame, self.fighter_id, best.track_id, None))
        self.fighter_id = best.track_id
        return best

    # --- kit identification ----------------------------------------------

    def _looks_like_fighter(self, dists: tuple[float, float] | None) -> bool:
        return dists is not None and dists[1] - dists[0] >= self.margin and dists[0] <= self.max_fighter_dist

    def _looks_like_opponent(self, dists: tuple[float, float] | None) -> bool:
        return dists is not None and dists[0] - dists[1] >= self.margin

    def _identify(self, people: list[PersonPose], frame: np.ndarray) -> PersonPose | None:
        plausible = [p for p in people if p.track_id is not None and p.bbox is not None
                     and (not self._heights and self._last_bbox is None or self._similar_size(p.bbox))]
        signatures = {p.track_id: kit_signature(frame, p) for p in plausible}
        dists = {tid: self.kit.distances(sig) for tid, sig in signatures.items()}

        opponent_height = max((_height(p) for p in plausible if self._looks_like_opponent(dists[p.track_id])),
                              default=None)
        fighters = [p for p in plausible
                    if p.track_id != self.fighter_id and self._looks_like_fighter(dists[p.track_id])
                    and (opponent_height is None or _height(p) >= self.min_opponent_height * opponent_height)]
        best = min(fighters, default=None, key=lambda p: dists[p.track_id][0] - dists[p.track_id][1])

        followed = select_fighter(people, self.fighter_id)
        chosen = None
        if followed is not None:
            followed_dists = dists.get(self.fighter_id)
            if followed_dists is None and followed.track_id not in dists:
                followed_dists = self.kit.distances(kit_signature(frame, followed))
            if best is not None and not self._looks_like_fighter(followed_dists):
                same = self._swap_candidate == best.track_id
                self._swap_candidate, self._swap_streak = best.track_id, self._swap_streak + 1 if same else 1
            else:
                self._swap_candidate, self._swap_streak = None, 0
            self._doubt_streak = self._doubt_streak + 1 if self._looks_like_opponent(followed_dists) else 0

            if self._swap_streak >= self.confirm_frames:
                self.swaps.append((self._frame, self.fighter_id, best.track_id,
                                   None if followed_dists is None else followed_dists[0], dists[best.track_id][0]))
                self.fighter_id = best.track_id
                self._swap_candidate, self._swap_streak, self._doubt_streak = None, 0, 0
                chosen = best
            elif self._doubt_streak < self.confirm_frames:
                chosen = followed
        elif self.relink and best is not None:
            self.relinks.append((self._frame, self.fighter_id, best.track_id, dists[best.track_id][0]))
            self.fighter_id = best.track_id
            chosen = best

        self._adapt(chosen, people, plausible, signatures, dists)
        return chosen

    def _adapt(self, chosen: PersonPose | None, people: list[PersonPose], plausible: list[PersonPose],
               signatures: dict[int, Signature], dists: dict[int, tuple[float, float] | None]) -> None:
        """Let both kit prototypes follow lighting changes, from confident,
        non-overlapping frames only."""
        for person in plausible:
            d = dists.get(person.track_id)
            if d is None or not self._is_clean(person, people):
                continue
            if person is chosen and d[1] - d[0] >= 2 * self.margin:
                self.kit.adapt("fighter", signatures[person.track_id], self.adapt_alpha)
            elif person is not chosen and d[0] - d[1] >= 2 * self.margin:
                self.kit.adapt("opponent", signatures[person.track_id], self.adapt_alpha)

    # --- geometry ----------------------------------------------------------

    def _is_clean(self, fighter: PersonPose, people: list[PersonPose]) -> bool:
        """Fighter fully on their own: overlapping no one else's box."""
        if fighter.bbox is None:
            return False
        return all(bbox_iou(fighter.bbox, p.bbox) < self.clean_iou
                   for p in people if p is not fighter and p.bbox is not None)

    def _reference_height(self) -> float:
        """Median height over recent clean frames, else the last box's height."""
        if self._heights:
            return float(np.median(self._heights))
        return self._last_bbox[3] - self._last_bbox[1]

    def _distance(self, bbox) -> float:
        """Center-to-center distance from the last box, in reference heights."""
        lx1, ly1, lx2, ly2 = self._last_bbox
        x1, y1, x2, y2 = bbox
        dx = ((x1 + x2) - (lx1 + lx2)) / 2 * self.aspect
        dy = ((y1 + y2) - (ly1 + ly2)) / 2
        return (dx * dx + dy * dy) ** 0.5 / self._reference_height()

    def _similar_size(self, bbox) -> bool:
        lo, hi = self.size_ratio
        ratio = (bbox[3] - bbox[1]) / self._reference_height()
        return lo <= ratio <= hi
