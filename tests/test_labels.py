import pytest

from pipeline.labels import (WINDOW, PunchLabel, auc, evaluate, load_guard_labels, load_punch_labels, parse_time,
                             pick_threshold, punch_kind, punch_window, shared_frames)


def split(downs: list[float], ups: list[float]):
    return pick_threshold(downs + ups, [True] * len(downs) + [False] * len(ups))


def test_clean_split_lands_between_the_groups():
    s = split([-1.0, -0.9], [-0.5, 0.0])
    assert s.threshold == pytest.approx(-0.7)
    assert s.balanced_accuracy == 1.0


def test_overlapping_groups_get_the_best_balanced_accuracy():
    s = split([-1.0, -0.8, -0.4], [-0.6, -0.2, 0.0])
    assert s.threshold == pytest.approx(-0.7)  # misses only the down at -0.4
    assert s.balanced_accuracy == pytest.approx(5 / 6)
    for t in (-1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1):
        assert evaluate(t, [-1.0, -0.8, -0.4, -0.6, -0.2, 0.0], [True] * 3 + [False] * 3).balanced_accuracy \
            <= s.balanced_accuracy + 1e-9


def test_tie_goes_to_the_widest_gap():
    # -0.85 (gap 0.3) and -0.2 (gap 0.4) both catch half of one group and all of the other
    s = split([-1.0, -0.4], [-0.7, 0.0])
    assert s.threshold == pytest.approx(-0.2)


def test_bigger_group_does_not_drag_the_threshold():
    # Plain accuracy would put it below -1.0 or at -0.95, giving up on the downs;
    # balanced accuracy keeps both downs and loses two outlying ups.
    ups = [-0.9, -0.85, -0.6, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    s = split([-1.0, -0.8], ups)
    assert s.threshold == pytest.approx(-0.7)
    assert (s.downs_caught, s.ups_caught) == (2, 8)


def test_needs_both_groups():
    with pytest.raises(ValueError):
        split([-1.0, -0.5], [])


def test_load_leaves_out_unsure_and_punch_but_counts_them(tmp_path):
    path = tmp_path / "guard_labels.csv"
    path.write_text("id,clip,time_s,frame,fighter,left_hand,right_hand\n"
                    "G001,clipA,3.90,117,fighter_1,punch,unsure\n"
                    "G002,clipA,12.07,362,fighter_2,down,UP\n", encoding="utf-8")
    labels, answers = load_guard_labels(path)
    assert [(l.id, l.frame, l.fighter, l.hand, l.label) for l in labels] == [
        ("G002", 362, "fighter_2", "left", "down"), ("G002", 362, "fighter_2", "right", "up")]
    assert answers == {"punch": 1, "unsure": 1, "down": 1, "up": 1}


def test_parse_time():
    assert parse_time("00:21") == 21.0
    assert parse_time("1:35.36") == pytest.approx(95.36)
    assert parse_time("12.5") == 12.5


def test_punch_kind_from_notes():
    assert punch_kind("1") == "1"
    assert punch_kind("2 to the body") == "2"
    assert punch_kind("second of 1-2") == "2"
    assert punch_kind("third of 1-2-1") == "1"
    assert punch_kind("") is None and punch_kind("feint") is None


def test_load_punch_labels_maps_kit_to_fighter_and_skips_unknown(tmp_path):
    path = tmp_path / "punch_labels.csv"
    path.write_text("clip,time_s,fighter,hand,view,notes\n"
                    "Recording 2026-09-24 104715,1:35.36,red headgear,right_hand,side,1\n"
                    "clinch_clip,00:21,red headgear,left_hand,back,1\n"
                    "clinch_clip,00:13,White Headgear,left_hand,side,second of 1-2\n", encoding="utf-8")
    labels, skipped = load_punch_labels(path)
    assert [(l.fighter, l.hand, l.time, l.exact, l.kind) for l in labels] == [
        ("fighter_2", "right", pytest.approx(95.36), True, "1"), ("fighter_1", "left", 13.0, False, "2")]
    assert len(skipped) == 1


def test_punch_windows_all_the_same_length():
    exact = PunchLabel("c", "1:00.50", 60.5, True, "fighter_1", "left", "side", "1")
    whole = PunchLabel("c", "1:00", 60.0, False, "fighter_1", "left", "side", "1")
    for lab in (exact, whole):
        a, b = punch_window(lab)
        assert b - a == pytest.approx(WINDOW)
    assert punch_window(exact)[0] < 60.5 < punch_window(exact)[1]
    assert punch_window(whole)[0] < 60.0 and punch_window(whole)[1] > 61.0  # covers the whole second


def test_auc():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1], [2]) == 0.0
    assert auc([2], [2]) == 0.5


def test_shared_frames_when_both_follow_one_track():
    a = {"frames": [{"frame": 0, "found": True, "track_id": 1}, {"frame": 1, "found": True, "track_id": 1}]}
    b = {"frames": [{"frame": 0, "found": True, "track_id": 2}, {"frame": 1, "found": True, "track_id": 1}]}
    assert shared_frames({"f1": a, "f2": b}) == {1}
    assert shared_frames({"f1": a}) == set()
