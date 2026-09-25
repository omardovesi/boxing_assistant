"""Helpers for picking one tracked fighter out of each frame's detections."""
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
