import numpy as np

from pipeline.common.keypoints import PersonPose
from pipeline.tracking import FighterFollower, bbox_iou, select_fighter, track_ids
from tests.helpers import body_keypoints, paint_kit


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


FIGHTER_BOX = (0.40, 0.20, 0.50, 0.80)
NEARBY_BOX = (0.45, 0.20, 0.55, 0.80)
FAR_BOX = (0.90, 0.20, 1.00, 0.80)


def test_follower_returns_fighter_while_id_present():
    f = FighterFollower(1)
    a = PersonPose(track_id=1, bbox=FIGHTER_BOX)
    assert f.update([a, PersonPose(track_id=2, bbox=FAR_BOX)]) is a
    assert f.relinks == []


def test_follower_relinks_to_new_id_nearby():
    f = FighterFollower(1)
    f.update([PersonPose(track_id=1, bbox=FIGHTER_BOX)])
    assert f.update([]) is None  # lost
    new = PersonPose(track_id=7, bbox=NEARBY_BOX)
    assert f.update([new]) is new
    assert f.fighter_id == 7
    assert f.relinks == [(2, 1, 7, None)]  # no frames given, so colours weren't checked


def test_follower_keeps_new_id_after_relink():
    f = FighterFollower(1)
    f.update([PersonPose(track_id=1, bbox=FIGHTER_BOX)])
    f.update([PersonPose(track_id=7, bbox=NEARBY_BOX)])
    again = PersonPose(track_id=7, bbox=NEARBY_BOX)
    assert f.update([again]) is again
    assert len(f.relinks) == 1


def test_follower_never_takes_opponents_existing_id():
    f = FighterFollower(1)
    opponent = PersonPose(track_id=2, bbox=NEARBY_BOX)
    f.update([PersonPose(track_id=1, bbox=FIGHTER_BOX), opponent])
    assert f.update([opponent]) is None
    assert f.fighter_id == 1


def test_follower_ignores_new_id_too_far_away():
    f = FighterFollower(1, aspect=16 / 9)  # 0.5 apart -> ~1.5 box-heights on 16:9
    f.update([PersonPose(track_id=1, bbox=FIGHTER_BOX)])
    assert f.update([PersonPose(track_id=7, bbox=FAR_BOX)]) is None


def test_follower_distance_accounts_for_frame_aspect():
    # 0.15 apart horizontally is 0.25 box-heights on a square frame but
    # ~0.44 box-heights on a 16:9 frame, so a 0.4 limit only rejects the latter.
    box = (0.55, 0.20, 0.65, 0.80)
    square = FighterFollower(1, aspect=1.0)
    square.update([PersonPose(track_id=1, bbox=FIGHTER_BOX)])
    assert square.update([PersonPose(track_id=7, bbox=box)]) is not None
    wide = FighterFollower(1, aspect=16 / 9, max_dist=0.4)
    wide.update([PersonPose(track_id=1, bbox=FIGHTER_BOX)])
    assert wide.update([PersonPose(track_id=7, bbox=box)]) is None


def test_follower_ignores_new_id_of_different_size():
    f = FighterFollower(1)
    f.update([PersonPose(track_id=1, bbox=FIGHTER_BOX)])
    small = (0.45, 0.60, 0.50, 0.80)  # a third of the height, e.g. a seated bystander
    assert f.update([PersonPose(track_id=7, bbox=small)]) is None


def test_follower_without_colours_gives_up_after_nearby_frames():
    f = FighterFollower(1, nearby_frames=3)
    f.update([PersonPose(track_id=1, bbox=FIGHTER_BOX)])
    for _ in range(4):
        f.update([])
    assert f.update([PersonPose(track_id=7, bbox=NEARBY_BOX)]) is None


def test_follower_picks_closest_of_several_new_ids():
    f = FighterFollower(1)
    f.update([PersonPose(track_id=1, bbox=FIGHTER_BOX)])
    closer = PersonPose(track_id=8, bbox=(0.41, 0.20, 0.51, 0.80))
    further = PersonPose(track_id=7, bbox=(0.52, 0.20, 0.62, 0.80))  # not overlapping `closer` (no clinch)
    assert f.update([further, closer]) is closer


def test_follower_cannot_relink_before_fighter_ever_seen():
    f = FighterFollower(1)
    assert f.update([PersonPose(track_id=7, bbox=NEARBY_BOX)]) is None


def test_bbox_iou():
    assert bbox_iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert bbox_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert bbox_iou((0, 0, 2, 1), (1, 0, 3, 1)) == 1 / 3


def test_follower_size_gate_uses_recent_height_not_shrinking_box():
    # The fighter disappears behind the opponent: his box shrinks to his legs
    # before he's lost. A small new ID nearby (e.g. a bystander) must not be
    # taken just because it matches that shrunken last box.
    f = FighterFollower(1)
    opponent = PersonPose(track_id=2, bbox=(0.38, 0.20, 0.55, 0.80))
    for _ in range(5):
        f.update([PersonPose(track_id=1, bbox=FIGHTER_BOX)])
    for _ in range(3):
        f.update([PersonPose(track_id=1, bbox=(0.40, 0.60, 0.50, 0.80)), opponent])
    f.update([opponent])  # lost
    small = PersonPose(track_id=7, bbox=(0.42, 0.60, 0.52, 0.80))
    assert f.update([opponent, small]) is None
    assert f.relinks == []


# Kit identification: people are painted as blocks of colour on a grey frame.
# Fighter: red headgear; opponent: blue headgear; same white shirts and black
# trunks, so the headgear is what tells them apart.
RED, BLUE, WHITE, BLACK = (0, 0, 255), (255, 0, 0), (255, 255, 255), (20, 20, 20)  # BGR
OPPONENT_BOX = FAR_BOX  # far enough from FIGHTER_BOX that neither overlaps the other
FIGHTER_KIT = dict(head=RED, shirt=WHITE, trunks=BLACK)
OPPONENT_KIT = dict(head=BLUE, shirt=WHITE, trunks=BLACK)


def _kitted(track_id: int, bbox) -> PersonPose:
    return PersonPose(keypoints=body_keypoints(bbox), bbox=bbox, track_id=track_id)


def _scene(*people_kits) -> np.ndarray:
    frame = np.full((400, 400, 3), 128, dtype=np.uint8)
    for person, kit in people_kits:
        paint_kit(frame, person, **kit)
    return frame


def _calibrated(**kwargs) -> tuple[FighterFollower, PersonPose, PersonPose]:
    """Fighter ID 1 and opponent ID 2 seen apart for 5 frames -> kit comparison done."""
    f = FighterFollower(1, calibration_frames=5, **kwargs)
    fighter, opponent = _kitted(1, FIGHTER_BOX), _kitted(2, OPPONENT_BOX)
    frame = _scene((fighter, FIGHTER_KIT), (opponent, OPPONENT_KIT))
    for _ in range(5):
        assert f.update([fighter, opponent], frame) is fighter  # by ID while calibrating
    return f, fighter, opponent


def test_follower_calibrates_on_the_kit_item_that_differs():
    f, _, _ = _calibrated()
    assert f.opponent_id == 2
    assert f.kit.weights == {"headgear": 1.0}


def test_follower_without_opponent_in_view_does_not_calibrate():
    f = FighterFollower(1, calibration_frames=5)
    fighter = _kitted(1, FIGHTER_BOX)
    for _ in range(10):
        assert f.update([fighter], _scene((fighter, FIGHTER_KIT))) is fighter
    assert f.kit is None


def test_follower_corrects_an_id_swap_after_confirm_frames():
    f, _, _ = _calibrated()
    # The tracker swapped IDs: ID 1 is now on the opponent, ID 2 on the fighter.
    id1_on_opponent, id2_on_fighter = _kitted(1, OPPONENT_BOX), _kitted(2, FIGHTER_BOX)
    frame = _scene((id1_on_opponent, OPPONENT_KIT), (id2_on_fighter, FIGHTER_KIT))
    for _ in range(f.confirm_frames - 1):
        assert f.update([id1_on_opponent, id2_on_fighter], frame) is id1_on_opponent  # not sure yet
    assert f.update([id1_on_opponent, id2_on_fighter], frame) is id2_on_fighter
    assert f.fighter_id == 2
    frame_idx, old, new, old_dist, new_dist = f.swaps[0]
    assert (old, new) == (1, 2)
    assert new_dist < old_dist


def test_follower_ignores_a_single_odd_frame():
    f, fighter, opponent = _calibrated()
    id1_on_opponent, id2_on_fighter = _kitted(1, OPPONENT_BOX), _kitted(2, FIGHTER_BOX)
    f.update([id1_on_opponent, id2_on_fighter],
             _scene((id1_on_opponent, OPPONENT_KIT), (id2_on_fighter, FIGHTER_KIT)))
    normal = _scene((fighter, FIGHTER_KIT), (opponent, OPPONENT_KIT))
    for _ in range(10):
        assert f.update([fighter, opponent], normal) is fighter
    assert f.swaps == []


def test_follower_relinks_to_whoever_wears_the_fighters_kit():
    f, _, opponent = _calibrated()
    new = _kitted(7, NEARBY_BOX)
    assert f.update([new, opponent], _scene((new, FIGHTER_KIT), (opponent, OPPONENT_KIT))) is new
    frame_idx, old, new_id, kit_dist = f.relinks[0]
    assert (old, new_id) == (1, 7)
    assert kit_dist < 0.2


def test_follower_never_relinks_to_someone_in_the_opponents_kit():
    f, _, _ = _calibrated()
    lookalike = _kitted(7, NEARBY_BOX)
    for _ in range(20):
        assert f.update([lookalike], _scene((lookalike, OPPONENT_KIT))) is None
    assert f.fighter_id == 1


def test_follower_ignores_a_smaller_bystander_in_the_fighters_kit():
    # e.g. someone in the crowd wearing similar colours, further from the camera
    f, _, opponent = _calibrated()
    bystander = _kitted(7, (0.45, 0.44, 0.55, 0.80))  # 0.6x the opponent's height
    frame = _scene((bystander, FIGHTER_KIT), (opponent, OPPONENT_KIT))
    assert f.update([bystander, opponent], frame) is None


def test_follower_reports_missing_when_followed_id_looks_like_the_opponent():
    f, _, _ = _calibrated()
    wrong = _kitted(1, FIGHTER_BOX)
    frame = _scene((wrong, OPPONENT_KIT))
    results = [f.update([wrong], frame) for _ in range(8)]
    assert results[:f.confirm_frames - 1] == [wrong] * (f.confirm_frames - 1)
    assert all(r is None for r in results[f.confirm_frames - 1:])


def test_follower_with_relink_off_never_takes_a_new_id():
    f, _, opponent = _calibrated(relink=False)
    new = _kitted(7, NEARBY_BOX)
    assert f.update([new, opponent], _scene((new, FIGHTER_KIT), (opponent, OPPONENT_KIT))) is None
    assert f.fighter_id == 1


# Clinches: while the fighter overlaps the opponent by clinch_iou (0.3) or
# more, nobody is reported; afterwards the fighter is re-identified by kit.
CLINCH_BOX = (0.43, 0.20, 0.53, 0.80)  # IoU ~0.54 with FIGHTER_BOX
# Just stepped out of the clinch: next to FIGHTER_BOX, not overlapping, within
# clinch_partner_dist (0.25 fighter heights between centres).
APART_BOX = (0.55, 0.20, 0.65, 0.80)


def _clinch(f, fighter_id=1, opponent_id=2):
    a, b = _kitted(fighter_id, FIGHTER_BOX), _kitted(opponent_id, CLINCH_BOX)
    return f.update([a, b], _scene((a, FIGHTER_KIT), (b, OPPONENT_KIT)))


def _apart(f, fighter_id=1, opponent_id=2):
    """Returns (result, the person wearing the fighter's kit)."""
    a, b = _kitted(fighter_id, FIGHTER_BOX), _kitted(opponent_id, APART_BOX)
    return f.update([a, b], _scene((a, FIGHTER_KIT), (b, OPPONENT_KIT))), a


def test_clinch_reports_fighter_missing():
    f, _, _ = _calibrated()
    assert _clinch(f) is None
    assert f.in_clinch and f.clinches == 1


def test_clinch_ends_after_exit_frames_and_reidentifies_same_id():
    f, _, _ = _calibrated()
    _clinch(f)
    for _ in range(f.clinch_exit_frames - 1):
        assert _apart(f)[0] is None  # still counted as the clinch
        assert f.in_clinch
    result, fighter = _apart(f)
    assert result is fighter and not f.in_clinch
    assert [(old, new) for _, old, new, _ in f.reidentified] == [(1, 1)]


def test_brief_separation_does_not_end_clinch():
    f, _, _ = _calibrated()
    _clinch(f)
    _apart(f)
    _clinch(f)
    assert f.in_clinch and f.clinches == 1


def test_after_clinch_follows_fighters_kit_when_tracker_swapped_ids():
    f, _, _ = _calibrated()
    _clinch(f)
    for _ in range(f.clinch_exit_frames - 1):
        _apart(f, fighter_id=2, opponent_id=1)
    # Tracker swapped: ID 2 now wears the fighter's kit. No confirm-frames delay.
    result, fighter = _apart(f, fighter_id=2, opponent_id=1)
    assert result is fighter and f.fighter_id == 2
    assert [(old, new) for _, old, new, _ in f.reidentified] == [(1, 2)]
    assert f.swaps == []


def test_after_clinch_missing_until_someone_wears_fighters_kit():
    f, _, _ = _calibrated()
    _clinch(f)
    # Clinch ends (two people seen apart) but neither wears the fighter's kit:
    # the followed ID is on the opponent, and an unpainted person is unclear.
    opponent, unclear = _kitted(1, APART_BOX), _kitted(5, FIGHTER_BOX)
    frame = _scene((opponent, OPPONENT_KIT))
    for _ in range(f.clinch_exit_frames + 5):
        assert f.update([opponent, unclear], frame) is None
    assert not f.in_clinch and f.reidentified == []
    result, fighter = _apart(f, fighter_id=7, opponent_id=1)
    assert result is fighter and f.fighter_id == 7


def test_clinch_detected_when_fighters_id_lost_inside_it():
    f, _, _ = _calibrated()
    _clinch(f)
    assert _clinch(f, fighter_id=7) is None  # ID 1 gone; 7 and 2 overlap where the fighter was
    assert f.in_clinch and f.clinches == 1
    for _ in range(f.clinch_exit_frames - 1):
        _apart(f, fighter_id=7)
    result, fighter = _apart(f, fighter_id=7)
    assert result is fighter
    assert [(old, new) for _, old, new, _ in f.reidentified] == [(1, 7)]


def test_small_bystander_overlapping_is_not_a_clinch():
    f, _, opponent = _calibrated()
    fighter = _kitted(1, FIGHTER_BOX)
    bystander = _kitted(7, (0.42, 0.50, 0.48, 0.70))  # a third of the fighter's height
    frame = _scene((fighter, FIGHTER_KIT), (opponent, OPPONENT_KIT))
    assert f.update([fighter, bystander, opponent], frame) is fighter
    assert not f.in_clinch


def test_clinch_before_calibration_resumes_by_track_id():
    f = FighterFollower(1)
    a, b = PersonPose(track_id=1, bbox=FIGHTER_BOX), PersonPose(track_id=2, bbox=CLINCH_BOX)
    assert f.update([a, b]) is None and f.in_clinch
    apart = [PersonPose(track_id=1, bbox=FIGHTER_BOX), PersonPose(track_id=2, bbox=APART_BOX)]
    results = [f.update(apart) for _ in range(f.clinch_exit_frames)]
    assert results[-1] is apart[0] and not f.in_clinch
    assert f.reidentified == []


def test_clinch_continues_while_one_fighter_hidden_behind_the_other():
    # clinch_clip frame 2584: only one merged box is detected at the fighter's
    # spot, so there's no overlap to measure. That's not "apart".
    f, _, _ = _calibrated()
    _clinch(f)
    merged = _kitted(2, FIGHTER_BOX)
    frame = _scene((merged, FIGHTER_KIT))  # would pass as the fighter if re-identified
    for _ in range(3 * f.clinch_exit_frames):
        assert f.update([merged], frame) is None
    assert f.in_clinch and f.reidentified == []
    for _ in range(f.clinch_exit_frames - 1):
        assert _apart(f)[0] is None
    result, fighter = _apart(f)
    assert result is fighter
    assert [(old, new) for _, old, new, _ in f.reidentified] == [(1, 1)]


def test_clinch_times_out_when_only_one_person_stays():
    f, _, _ = _calibrated(clinch_timeout=10)
    _clinch(f)
    alone = _kitted(1, FIGHTER_BOX)
    frame = _scene((alone, FIGHTER_KIT))
    results = [f.update([alone], frame) for _ in range(10)]
    assert all(r is None for r in results[:-1])
    assert results[-1] is alone and not f.in_clinch


def test_person_far_away_does_not_end_clinch():
    # e.g. the coach at ringside while one fighter hides the other
    f, _, _ = _calibrated()
    _clinch(f)
    merged, coach = _kitted(1, FIGHTER_BOX), _kitted(9, FAR_BOX)  # coach ~0.83 fighter heights away
    frame = _scene((merged, FIGHTER_KIT), (coach, OPPONENT_KIT))
    for _ in range(3 * f.clinch_exit_frames):
        assert f.update([merged, coach], frame) is None
    assert f.in_clinch
