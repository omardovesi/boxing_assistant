import pytest

from pipeline.common.keypoints import COCO17_NAMES
from pipeline.signals import baseline, elbow_opening, extension_above_baseline, wrist_speed
from tests.helpers import body_keypoints

FPS = 30.0


def frame(left_wrist=(0.2, 0.5), overlap=0.0, found=True) -> dict:
    """Standing pose filling the frame (torso 0.3; left shoulder (0.2, 0.25),
    left elbow (0.15, 0.4)) with the left wrist moved to `left_wrist`."""
    if not found:
        return {"found": False}
    kps = [[k.x, k.y, k.confidence] for k in body_keypoints((0.0, 0.0, 1.0, 1.0))]
    kps[COCO17_NAMES.index("left_wrist")] = [*left_wrist, 0.9]
    return {"found": True, "overlap": overlap, "keypoints": kps}


def test_baseline_ignores_untrusted_frames():
    assert baseline([1.0, 2.0, 3.0, 100.0], [1.0, 1.0, 1.0, 0.2]) == 2.0
    assert baseline([None, 5.0], [1.0, 0.0]) is None


def test_extension_measured_from_the_hands_usual_length():
    # resting wrist 0.25 below the shoulder (0.83 torso lengths), then 5 frames
    # punching out level with the shoulder, 0.33 away (1.1 torso lengths)
    rest, out = (0.2, 0.5), (0.53, 0.25)
    frames = [frame(rest)] * 20 + [frame(out)] * 5 + [frame(rest)] * 20
    ext = extension_above_baseline(frames, 1.0, FPS, "left")
    assert ext[0] == pytest.approx(0.0)
    assert max(ext) > 0.25


def test_wrist_speed_zero_when_still_and_high_when_the_wrist_moves():
    frames = [frame()] * 10 + [frame((0.2 - 0.01 * i, 0.5)) for i in range(1, 11)]
    speed = wrist_speed(frames, 1.0, FPS, "left")
    assert speed[:3] == [None, None, None]  # nothing 0.1 s earlier to compare with
    assert speed[8] == pytest.approx(0.0)
    # 0.01 per frame = 0.3 per second = 1 torso length per second
    assert speed[19] == pytest.approx(1.0, rel=0.05)


def test_wrist_speed_needs_both_ends_of_its_span():
    # 0.1 s = 3 frames: frame i compares with frame i - 3
    frames = [frame()] * 5 + [frame(found=False)] + [frame()] * 5
    speed = wrist_speed(frames, 1.0, FPS, "left")
    assert speed[5] is None and speed[8] is None  # an end is missing
    assert speed[6] == pytest.approx(0.0)  # the gap is only in between


def test_elbow_opening_counts_degrees_gained_recently():
    bent = (0.3, 0.4)  # forearm pointing forward: about 90 degrees at the elbow
    straight = (0.10, 0.55)  # in line with the upper arm: 180
    frames = [frame(bent)] * 10 + [frame(straight)] * 10
    opening = elbow_opening(frames, 1.0, FPS, "left")
    assert opening[5] == pytest.approx(0.0)
    assert opening[11] > 60
    assert opening[19] == pytest.approx(0.0)  # held straight: no longer opening
