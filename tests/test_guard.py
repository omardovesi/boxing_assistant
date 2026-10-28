import pytest

from pipeline.guard import detect_all, detect_guard_drops, low_stretches, punch_zone
from tests.test_signals import FPS, frame

# Left shoulder at y 0.25, torso 0.3: a wrist at y 0.5 reads -0.83 (the usual
# guard here); each 0.03 lower is 0.1 torso lengths lower.
USUAL_Y = 0.5


def clip(drop_y: float, drop_frames: int, rest_frames: int = 60, overlap: float = 0.0) -> list[dict]:
    """Guard at USUAL_Y, then the left hand at drop_y for drop_frames, then back."""
    rest = [frame((0.2, USUAL_Y))] * rest_frames
    return rest + [frame((0.2, drop_y), overlap=overlap)] * drop_frames + rest


def test_low_stretches_need_the_minimum_length():
    below = [False] + [True] * 9 + [False] + [True] * 8 + [False]
    assert low_stretches(below, FPS, min_seconds=0.3) == [(1, 9)]  # 9 frames = 0.3 s; 8 is too short


def test_punch_zone_covers_the_punch_and_the_time_after():
    zone = punch_zone(30, FPS, [{"start": 0.1, "end": 0.2}], after=0.1)
    assert zone[2] is False and zone[3] and zone[9] and zone[10] is False


def test_long_drop_below_usual_is_an_event():
    events = detect_guard_drops(clip(USUAL_Y + 0.15, 15), 1.0, FPS, "left", [])
    assert len(events) == 1
    e = events[0]
    assert e["start"] == pytest.approx(60 / FPS) and e["duration"] == pytest.approx(15 / FPS)
    assert e["usual"] == pytest.approx(-0.83, abs=0.01) and e["below_usual"] == pytest.approx(0.5, abs=0.01)


def test_short_dip_is_not_a_drop():
    assert detect_guard_drops(clip(USUAL_Y + 0.15, 6), 1.0, FPS, "left", []) == []


def test_small_drop_is_not_a_drop():
    # 0.15 torso lengths below usual: within the margin
    assert detect_guard_drops(clip(USUAL_Y + 0.045, 30), 1.0, FPS, "left", []) == []


def test_drop_during_or_just_after_a_punch_is_ignored():
    punch = {"start": 55 / FPS, "end": 62 / FPS}  # hand still coming back afterwards
    assert detect_guard_drops(clip(USUAL_Y + 0.15, 15), 1.0, FPS, "left", [punch]) == []


def test_low_guard_style_only_fires_when_lower_still():
    low = USUAL_Y + 0.12  # carries the hand 0.4 lower than the others all the time
    steady = [frame((0.2, low))] * 120
    assert detect_guard_drops(steady, 1.0, FPS, "left", []) == []
    lower = [frame((0.2, low))] * 60 + [frame((0.2, low + 0.15))] * 15 + [frame((0.2, low))] * 60
    assert len(detect_guard_drops(lower, 1.0, FPS, "left", [])) == 1


def test_untrusted_frames_dont_make_a_drop():
    assert detect_guard_drops(clip(USUAL_Y + 0.15, 15, overlap=0.5), 1.0, FPS, "left", []) == []


def test_detect_all_gives_each_hand_only_its_own_punches():
    data = {"width": 1, "height": 1, "fps": FPS, "frames": clip(USUAL_Y + 0.15, 15)}
    right_punch = {"hand": "right", "start": 55 / FPS, "end": 62 / FPS}
    events = detect_all(data, [right_punch])
    assert [e["hand"] for e in events] == ["left"]  # a right-hand punch doesn't excuse the left hand
