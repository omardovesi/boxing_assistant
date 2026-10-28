from pathlib import Path

import pytest

from pipeline.summary import build_summary, frame_repeats, output_path
from tests.test_punches import jab
from tests.test_signals import FPS, frame

GUARD_UP = (0.2, 0.2)  # left wrist above its shoulder (y 0.25): guard height +0.17


def data(frames: list[dict]) -> dict:
    return {"video": "clip.mp4", "fighter_id": 1, "fps": FPS, "width": 1, "height": 1,
            "frames": frames, "clinches": [{"start": 0, "end": 1}]}


def session() -> list[dict]:
    """A jab, then the left hand dropping for half a second, then rest."""
    return ([frame()] * 30 + jab() + [frame()] * 60
            + [frame((0.2, 0.65))] * 15 + [frame()] * 30)


def test_events_of_both_types_sorted_by_time():
    s = build_summary(data(session()))
    assert [e["type"] for e in s["events"]] == ["punch", "guard_drop"]
    times = [e["time"] for e in s["events"]]
    assert times == sorted(times)
    assert s["events"][0]["time"] == s["events"][0]["peak"] and s["events"][1]["time"] == s["events"][1]["start"]


def test_counts_match_the_events():
    s = build_summary(data(session()))
    assert s["punches"]["total"] == s["punches"]["left"] == 1 and s["punches"]["right"] == 0
    assert s["guard"]["left"]["drops"] == 1 and s["guard"]["right"]["drops"] == 0
    minutes = len(session()) / FPS / 60  # every frame trusted
    assert s["usable_minutes"] == pytest.approx(minutes, abs=0.001)
    assert s["punches"]["per_usable_minute"] == pytest.approx(1 / minutes, abs=0.01)
    assert s["clinches"] == 1


def test_exposure_is_the_share_of_time_below_the_fixed_standard():
    low = build_summary(data([frame()] * 60))  # wrist 0.83 below the shoulder line
    high = build_summary(data([frame(GUARD_UP)] * 60))
    assert low["guard"]["left"]["exposure"] == {"value": 1.0, "coverage": 1.0}
    assert high["guard"]["left"]["exposure"]["value"] == 0.0
    half = build_summary(data([frame()] * 30 + [frame(GUARD_UP)] * 30))
    assert half["guard"]["left"]["exposure"]["value"] == pytest.approx(0.5, abs=0.05)


def test_untrusted_frames_lower_coverage_not_the_value():
    frames = [frame()] * 30 + [frame(GUARD_UP, overlap=0.5)] * 30  # second half weight 0
    s = build_summary(data(frames))
    assert s["guard"]["left"]["exposure"] == {"value": 1.0, "coverage": 0.5}


def test_guard_height_per_minute():
    frames = [frame()] * round(60 * FPS) + [frame(GUARD_UP)] * round(30 * FPS)
    per_minute = build_summary(data(frames))["guard"]["left"]["height_per_minute"]
    assert [m["minute"] for m in per_minute] == [0, 1]
    assert per_minute[0]["value"] == pytest.approx(-0.833, abs=0.01)
    assert per_minute[1]["value"] == pytest.approx(0.167, abs=0.01)


def test_repeat_flags_choose_the_guard_filter():
    frames = [frame()] * 10
    assert build_summary(data(frames))["guard_filter"] == "frames"
    assert build_summary(data(frames), repeats=[False] * 10)["guard_filter"] == "pictures"
    flagged = [{**f, "repeat": i % 2 == 1} for i, f in enumerate(frames)]
    assert frame_repeats(data(flagged), [False] * 10) == [i % 2 == 1 for i in range(10)]  # the JSON's own flags win


def test_output_sits_next_to_the_keypoints_file():
    assert output_path(Path("out/clip/fighter_1.json")) == Path("out/clip/fighter_1_metrics.json")
