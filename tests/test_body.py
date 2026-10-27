import pytest

from pipeline import config
from pipeline.body import arm_extension, elbow_angle, facing, guard_height, point, torso_length, wrist_offset
from pipeline.common.keypoints import COCO17_NAMES
from tests.helpers import body_keypoints

# Standing pose (tests/helpers.py _POSE) in a box filling the frame:
# shoulders at y 0.25 (x 0.2 / 0.8), hips at y 0.55 (x 0.3 / 0.7), wrists at y 0.5.
# Mid-shoulders (0.5, 0.25) to mid-hips (0.5, 0.55): torso length 0.3.
BOX = (0.0, 0.0, 1.0, 1.0)


def frame(bbox=BOX, **moved) -> dict:
    """A found JSON frame with the standing pose; `moved` overrides keypoints
    by name with [x, y, conf]."""
    kps = [[k.x, k.y, k.confidence] for k in body_keypoints(bbox)]
    for name, kp in moved.items():
        kps[COCO17_NAMES.index(name)] = kp
    return {"found": True, "keypoints": kps}


MISSING = {"found": False}


def test_point_scales_x_by_aspect():
    assert point(frame(), "left_shoulder", 2.0) == pytest.approx((0.4, 0.25))


def test_point_none_when_not_confident_or_not_found():
    weak = frame(left_wrist=[0.2, 0.5, config.MIN_KEYPOINT_CONF - 0.01])
    assert point(weak, "left_wrist", 1.0) is None
    assert point(MISSING, "left_wrist", 1.0) is None


def test_torso_length():
    assert torso_length(frame(), 1.0) == pytest.approx(0.3)


def test_torso_length_is_vertical_so_aspect_does_not_change_it():
    assert torso_length(frame(), 16 / 9) == pytest.approx(0.3)


def test_torso_length_none_without_a_hip():
    assert torso_length(frame(left_hip=[0.3, 0.55, 0.0]), 1.0) is None


def test_measurements_scale_with_distance_from_camera():
    # The same pose at half the size (fighter further away) measures the same.
    near, far = frame(), frame(bbox=(0.25, 0.25, 0.75, 0.75))
    assert guard_height(far, 1.0, "left") == pytest.approx(guard_height(near, 1.0, "left"))
    assert arm_extension(far, 1.0, "left") == pytest.approx(arm_extension(near, 1.0, "left"))
    assert facing(far, 1.0) == pytest.approx(facing(near, 1.0))


def test_guard_height_below_and_above_the_shoulder():
    # Wrist at y 0.5, shoulder at 0.25: 0.25 below, over a 0.3 torso.
    assert guard_height(frame(), 1.0, "left") == pytest.approx(-0.25 / 0.3)
    raised = frame(right_wrist=[0.8, 0.10, 0.9])
    assert guard_height(raised, 1.0, "right") == pytest.approx(0.15 / 0.3)


def test_guard_height_level_with_shoulder_is_zero():
    assert guard_height(frame(left_wrist=[0.1, 0.25, 0.9]), 1.0, "left") == pytest.approx(0.0)


def test_guard_height_none_without_wrist():
    assert guard_height(frame(left_wrist=[0.2, 0.5, 0.1]), 1.0, "left") is None
    assert guard_height(frame(left_wrist=[0.2, 0.5, 0.1]), 1.0, "right") is not None


def test_arm_extension():
    # Shoulder (0.2, 0.25) to wrist (0.2, 0.5): 0.25.
    assert arm_extension(frame(), 1.0, "left") == pytest.approx(0.25 / 0.3)
    punching = frame(left_wrist=[-0.1, 0.25, 0.9])  # arm straight out sideways, 0.3 away
    assert arm_extension(punching, 1.0, "left") == pytest.approx(1.0)


def test_arm_extension_uses_aspect():
    # Arm out sideways 0.1 frame widths: 0.2 frame heights on a 2:1 frame.
    sideways = frame(left_wrist=[0.1, 0.25, 0.9])
    assert arm_extension(sideways, 2.0, "left") == pytest.approx(0.2 / 0.3)


def test_facing_wide_shoulders_vs_side_on():
    assert facing(frame(), 1.0) == pytest.approx(0.6 / 0.3)
    side_on = frame(left_shoulder=[0.48, 0.25, 0.9], right_shoulder=[0.52, 0.25, 0.9])
    assert facing(side_on, 1.0) == pytest.approx(0.04 / 0.3)


def test_measurements_none_when_fighter_missing():
    for fn in (guard_height, arm_extension):
        assert fn(MISSING, 1.0, "left") is None
    assert facing(MISSING, 1.0) is None
    assert torso_length(MISSING, 1.0) is None


def test_elbow_angle_straight_and_bent():
    # left shoulder (0.2, 0.25), elbow (0.15, 0.4): wrist on the same line = straight arm
    straight = frame(left_wrist=[0.10, 0.55, 0.9])
    assert elbow_angle(straight, 1.0, "left") == pytest.approx(180.0)
    bent = frame(left_elbow=[0.2, 0.4, 0.9], left_wrist=[0.35, 0.4, 0.9])
    assert elbow_angle(bent, 1.0, "left") == pytest.approx(90.0)


def test_elbow_angle_none_without_wrist():
    assert elbow_angle(frame(left_wrist=[0.2, 0.5, 0.0]), 1.0, "left") is None


def test_wrist_offset_from_shoulder_in_torso_lengths():
    # left wrist (0.2, 0.5) is 0.25 below its shoulder; torso 0.3
    assert wrist_offset(frame(), 1.0, "left") == pytest.approx((0.0, 0.25 / 0.3))
    # moving the whole body doesn't change it
    shifted = frame(bbox=(0.1, 0.0, 1.1, 1.0))
    assert wrist_offset(shifted, 1.0, "left") == pytest.approx((0.0, 0.25 / 0.3))
