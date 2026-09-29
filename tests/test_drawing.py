import numpy as np

from pipeline.common.drawing import draw_id_labels, draw_skeleton
from pipeline.common.keypoints import COCO17_NAMES, Keypoint, PersonPose

FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


def _person(confidence: float) -> PersonPose:
    # Place every keypoint at the center with a uniform confidence.
    return PersonPose(keypoints=[Keypoint(x=0.5, y=0.5, confidence=confidence) for _ in COCO17_NAMES])


def test_returns_same_shape_as_input():
    out = draw_skeleton(FRAME, [_person(0.9)])
    assert out.shape == FRAME.shape


def test_does_not_mutate_input_frame():
    original = FRAME.copy()
    draw_skeleton(FRAME, [_person(0.9)])
    assert np.array_equal(FRAME, original)


def test_low_confidence_keypoints_are_not_drawn():
    high_conf = draw_skeleton(FRAME, [_person(0.9)], confidence_threshold=0.3)
    low_conf = draw_skeleton(FRAME, [_person(0.1)], confidence_threshold=0.3)
    assert not np.array_equal(high_conf, FRAME)
    assert np.array_equal(low_conf, FRAME)


def test_handles_no_people():
    out = draw_skeleton(FRAME, [])
    assert np.array_equal(out, FRAME)


def test_handles_multiple_people():
    out = draw_skeleton(FRAME, [_person(0.9), _person(0.9)])
    assert out.shape == FRAME.shape


def test_fixed_color_overrides_per_id_colors():
    red = (0, 0, 255)
    people = [PersonPose(keypoints=_person(0.9).keypoints, track_id=tid) for tid in (1, 2)]
    out = draw_skeleton(FRAME, people, color=red)
    drawn = out[np.any(out != 0, axis=2)]
    assert len(drawn) > 0
    assert all(tuple(int(c) for c in px) == red for px in drawn)


def test_id_labels_draw_on_a_copy():
    person = PersonPose(bbox=(0.2, 0.2, 0.8, 0.8), track_id=1)
    original = FRAME.copy()
    out = draw_id_labels(FRAME, [person])
    assert out.shape == FRAME.shape
    assert not np.array_equal(out, FRAME)
    assert np.array_equal(FRAME, original)


def test_id_labels_skip_untracked_people():
    person = PersonPose(bbox=(0.2, 0.2, 0.8, 0.8), track_id=None)
    assert np.array_equal(draw_id_labels(FRAME, [person]), FRAME)
