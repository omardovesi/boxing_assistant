import pytest

from pipeline.labels import evaluate, load_guard_labels, pick_threshold


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
