from pipeline.common.keypoints import PersonPose
from pipeline.tracking import select_fighter, track_ids


def test_select_fighter_returns_matching_person():
    a, b = PersonPose(track_id=1), PersonPose(track_id=2)
    assert select_fighter([a, b], 2) is b


def test_select_fighter_returns_none_when_id_missing():
    assert select_fighter([PersonPose(track_id=1)], 5) is None


def test_select_fighter_handles_no_people():
    assert select_fighter([], 1) is None


def test_select_fighter_ignores_untracked_people():
    assert select_fighter([PersonPose(track_id=None)], 1) is None


def test_track_ids_are_sorted_and_skip_untracked():
    people = [PersonPose(track_id=3), PersonPose(track_id=None), PersonPose(track_id=1)]
    assert track_ids(people) == [1, 3]


def test_track_ids_handles_no_people():
    assert track_ids([]) == []
