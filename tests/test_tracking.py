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
    assert f.update([a, PersonPose(track_id=2, bbox=NEARBY_BOX)]) is a
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
    assert f.update([PersonPose(track_id=7, bbox=NEARBY_BOX), closer]) is closer


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


# Clinch: the boxes overlap, so each person's kit crops contain the other's
# kit. Kit evidence from these frames must not move the follower.
CLINCH_BOX = (0.43, 0.20, 0.53, 0.80)  # IoU ~0.54 with FIGHTER_BOX


# Overlapping bodies (IoU ~0.13) but separate heads, so the headgear crops
# clearly show a swapped kit: only the overlap should stop the swap.
BODY_CLINCH_BOX = (0.47, 0.35, 0.57, 0.95)


def test_follower_does_not_swap_during_a_clinch():
    f, _, _ = _calibrated()
    # Looks exactly like an ID swap, but the two are overlapping.
    id1, id2 = _kitted(1, FIGHTER_BOX), _kitted(2, BODY_CLINCH_BOX)
    frame = _scene((id1, OPPONENT_KIT), (id2, FIGHTER_KIT))
    for _ in range(3 * f.confirm_frames):
        assert f.update([id1, id2], frame) is id1
    assert f.swaps == []
    assert f.fighter_id == 1


def test_follower_swap_streak_resumes_after_a_clinch():
    f, _, _ = _calibrated()
    id1_apart, id2_apart = _kitted(1, OPPONENT_BOX), _kitted(2, FIGHTER_BOX)
    apart = _scene((id1_apart, OPPONENT_KIT), (id2_apart, FIGHTER_KIT))
    id1_clinch, id2_clinch = _kitted(1, FIGHTER_BOX), _kitted(2, CLINCH_BOX)
    clinch = _scene((id1_clinch, OPPONENT_KIT), (id2_clinch, FIGHTER_KIT))
    for _ in range(f.confirm_frames - 1):
        f.update([id1_apart, id2_apart], apart)
    for _ in range(10):
        f.update([id1_clinch, id2_clinch], clinch)  # neither adds to nor resets the streak
    assert f.swaps == []
    assert f.update([id1_apart, id2_apart], apart) is id2_apart
    assert f.fighter_id == 2


def test_follower_does_not_relink_to_someone_in_a_clinch():
    f, _, _ = _calibrated()
    candidate, opponent = _kitted(7, CLINCH_BOX), _kitted(2, FIGHTER_BOX)
    frame = _scene((opponent, OPPONENT_KIT), (candidate, FIGHTER_KIT))
    for _ in range(5):
        assert f.update([candidate, opponent], frame) is None  # fighter's ID gone, candidate overlaps
    assert f.relinks == []
    candidate, opponent = _kitted(7, NEARBY_BOX), _kitted(2, OPPONENT_BOX)
    apart = _scene((candidate, FIGHTER_KIT), (opponent, OPPONENT_KIT))
    assert f.update([candidate, opponent], apart) is candidate
    assert f.fighter_id == 7


def test_follower_keeps_following_in_a_clinch_even_if_kit_looks_wrong():
    f, _, _ = _calibrated()
    followed, opponent = _kitted(1, FIGHTER_BOX), _kitted(2, CLINCH_BOX)
    frame = _scene((opponent, OPPONENT_KIT), (followed, OPPONENT_KIT))
    for _ in range(3 * f.confirm_frames):
        assert f.update([followed, opponent], frame) is followed
