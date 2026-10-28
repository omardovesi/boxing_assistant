from pipeline.punches import detect_punches, fast_stretches
from tests.test_signals import FPS, frame


def test_fast_stretches_merge_out_and_back():
    speed = [0, 9, 9, 0, 0, 9, 0] + [0] * 10 + [9, 0]
    usable = [True] * len(speed)
    # gap of 2 frames (0.07 s) merges; the late one is 0.33 s away
    assert fast_stretches(speed, usable, FPS, threshold=5, merge=0.2) == [(1, 5), (17, 17)]


def test_fast_stretches_skip_untrusted_and_missing_frames():
    speed = [9, None, 9, 9]
    usable = [True, True, False, True]
    assert fast_stretches(speed, usable, FPS, threshold=5, merge=0.0) == [(0, 0), (3, 3)]


def jab(rest=(0.2, 0.5), out=(0.53, 0.5), steps=4, hold=3):
    """Wrist moves out from rest (0.83 torso lengths from the shoulder) to
    `out` (1.38 by default), stays `hold` frames, and comes back."""
    path = [(rest[0] + (out[0] - rest[0]) * k / steps, rest[1] + (out[1] - rest[1]) * k / steps) for k in range(steps + 1)]
    return [frame(p) for p in path + [out] * (hold - 1) + path[::-1][1:]]


def test_a_jab_is_one_punch():
    frames = [frame()] * 30 + jab() + [frame()] * 30
    events = detect_punches(frames, 1.0, FPS, "left")
    assert len(events) == 1
    e = events[0]
    assert e["hand"] == "left" and 30 / FPS <= e["peak"] <= 39 / FPS
    assert e["extension"] > 0.2


def test_fast_but_short_movement_is_not_a_punch():
    # wrist snaps down and back up: fast, but the arm gets no longer than usual
    frames = [frame()] * 30 + jab(out=(0.2, 0.42)) + [frame()] * 30
    assert detect_punches(frames, 1.0, FPS, "left") == []


def test_no_punch_on_untrusted_frames():
    frames = [frame()] * 30 + [frame(p["keypoints"][9][:2], overlap=0.5) for p in jab()] + [frame()] * 30
    assert detect_punches(frames, 1.0, FPS, "left") == []
