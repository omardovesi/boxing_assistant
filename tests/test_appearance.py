import numpy as np

from pipeline.appearance import build_kit_profile, histogram_distance, kit_signature, region_boxes
from pipeline.common.keypoints import PersonPose
from tests.helpers import body_keypoints, paint_kit

BOX = (0.35, 0.25, 0.65, 0.90)
PERSON = PersonPose(keypoints=body_keypoints(BOX), bbox=BOX)
RED, BLUE, WHITE, BLACK = (0, 0, 255), (255, 0, 0), (255, 255, 255), (20, 20, 20)  # BGR


def _frame(**kit) -> np.ndarray:
    frame = np.full((400, 400, 3), 128, dtype=np.uint8)
    paint_kit(frame, PERSON, **kit)
    return frame


def test_all_kit_regions_found_for_a_visible_person():
    assert set(region_boxes((400, 400, 3), PERSON)) == {"headgear", "shirt", "trunks"}


def test_headgear_box_covers_the_top_of_the_head_not_just_the_face():
    x1, y1, x2, y2 = region_boxes((400, 400, 3), PERSON)["headgear"]
    face_top = min(PERSON.keypoints[i].y for i in range(5)) * 400
    assert y1 < face_top - 10  # reaches well above the eyes/ears


def test_region_with_unconfident_keypoints_is_skipped():
    keypoints = body_keypoints(BOX)
    for i in range(5):  # face points
        keypoints[i].confidence = 0.1
    boxes = region_boxes((400, 400, 3), PersonPose(keypoints=keypoints, bbox=BOX))
    assert "headgear" not in boxes
    assert "shirt" in boxes


def test_white_and_black_shirts_are_far_apart():
    # Colourless clothing still counts: this is what separates many sparring partners.
    white = kit_signature(_frame(shirt=WHITE), PERSON)["shirt"]
    black = kit_signature(_frame(shirt=BLACK), PERSON)["shirt"]
    assert histogram_distance(white, black) > 0.8


def test_same_headgear_in_dimmer_light_stays_close():
    bright = kit_signature(_frame(head=(0, 0, 255)), PERSON)["headgear"]
    dim = kit_signature(_frame(head=(0, 0, 150)), PERSON)["headgear"]
    red_vs_blue = histogram_distance(bright, kit_signature(_frame(head=BLUE), PERSON)["headgear"])
    assert histogram_distance(bright, dim) < red_vs_blue / 3


def _samples(head, n=6):
    frame = _frame(head=head, shirt=WHITE, trunks=BLACK)
    return [kit_signature(frame, PERSON) for _ in range(n)]


def test_kit_comparison_uses_only_what_differs():
    # Same shirts and trunks, different headgear: headgear decides alone.
    profile = build_kit_profile(_samples(RED), _samples(BLUE))
    assert profile.weights == {"headgear": 1.0}
    assert profile.report["shirt"][2] < 0.1
    assert "headgear" in profile.describe()


def test_kit_comparison_fails_when_nothing_differs():
    assert build_kit_profile(_samples(RED), _samples(RED)) is None


def test_kit_comparison_needs_enough_samples():
    assert build_kit_profile(_samples(RED, n=2), _samples(BLUE, n=2)) is None


def test_profile_distances_tell_fighter_from_opponent():
    profile = build_kit_profile(_samples(RED), _samples(BLUE))
    to_fighter, to_opponent = profile.distances(_samples(RED, n=1)[0])
    assert to_fighter < to_opponent
    to_fighter, to_opponent = profile.distances(_samples(BLUE, n=1)[0])
    assert to_opponent < to_fighter


def test_profile_distances_none_without_weighted_regions():
    profile = build_kit_profile(_samples(RED), _samples(BLUE))
    no_head = {k: v for k, v in _samples(RED, n=1)[0].items() if k != "headgear"}
    assert profile.distances(no_head) is None


def test_adapt_moves_prototype_only_part_way():
    profile = build_kit_profile(_samples(RED), _samples(BLUE))
    before = profile.fighter["headgear"].copy()
    profile.adapt("fighter", _samples(BLUE, n=1)[0], alpha=0.1)
    after = profile.fighter["headgear"]
    assert not np.array_equal(before, after)
    assert histogram_distance(after, before) < histogram_distance(after, profile.opponent["headgear"])
