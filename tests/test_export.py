import json

from pipeline.common.keypoints import Keypoint, PersonPose
from pipeline.export import CLINCH_IOU, clinch_overlap, frame_record, write_keypoints_json
from tests.helpers import body_keypoints

FIGHTER_BOX = (0.40, 0.20, 0.50, 0.80)
FAR_BOX = (0.80, 0.20, 0.90, 0.80)


def person(track_id, bbox):
    return PersonPose(keypoints=body_keypoints(bbox), bbox=bbox, track_id=track_id)


def shifted(bbox, dx):
    return (bbox[0] + dx, bbox[1], bbox[2] + dx, bbox[3])


def test_found_frame_has_all_fields():
    fighter = person(3, FIGHTER_BOX)
    r = frame_record(5, 0.16667, fighter, [fighter])
    assert r["frame"] == 5 and r["time"] == 0.1667 and r["found"] is True
    assert r["track_id"] == 3
    assert r["bbox"] == [0.4, 0.2, 0.5, 0.8]
    assert len(r["keypoints"]) == 17 and all(len(k) == 3 for k in r["keypoints"])
    assert r["overlap"] == 0.0 and r["clinch"] is False


def test_values_are_rounded_to_4_decimals():
    fighter = PersonPose(keypoints=[Keypoint(0.123456, 0.654321, 0.987654)] * 17,
                         bbox=(0.111111, 0.222222, 0.333333, 0.444444), track_id=1)
    r = frame_record(0, 0.0, fighter, [fighter])
    assert r["keypoints"][0] == [0.1235, 0.6543, 0.9877]
    assert r["bbox"] == [0.1111, 0.2222, 0.3333, 0.4444]


def test_missing_frame_has_only_frame_time_found():
    assert frame_record(7, 0.2333, None, []) == {"frame": 7, "time": 0.2333, "found": False}


def test_overlap_is_zero_without_opponent():
    fighter = person(1, FIGHTER_BOX)
    assert clinch_overlap(fighter, [fighter]) == 0.0


def test_overlap_is_zero_when_apart():
    fighter = person(1, FIGHTER_BOX)
    assert clinch_overlap(fighter, [fighter, person(2, FAR_BOX)]) == 0.0


def test_overlap_is_one_for_identical_boxes():
    fighter = person(1, FIGHTER_BOX)
    assert clinch_overlap(fighter, [fighter, person(2, FIGHTER_BOX)]) == 1.0


def test_clinch_flips_at_threshold():
    fighter = person(1, FIGHTER_BOX)
    # Same-size boxes shifted by dx overlap with IoU (w - dx) / (w + dx), w = 0.1.
    # IoU 0.3 exactly at dx = 0.7w / 1.3.
    at = 0.07 / 1.3
    inside = frame_record(0, 0.0, fighter, [fighter, person(2, shifted(FIGHTER_BOX, at - 0.001))])
    outside = frame_record(0, 0.0, fighter, [fighter, person(2, shifted(FIGHTER_BOX, at + 0.001))])
    assert inside["overlap"] >= CLINCH_IOU and inside["clinch"] is True
    assert outside["overlap"] < CLINCH_IOU and outside["clinch"] is False


def test_ignores_small_bystander():
    fighter = person(1, FIGHTER_BOX)
    bystander = person(2, (0.42, 0.50, 0.48, 0.70))  # inside the fighter's box, a third the height
    assert clinch_overlap(fighter, [fighter, bystander]) == 0.0


def test_written_file_loads_back(tmp_path):
    fighter = person(1, FIGHTER_BOX)
    frames = [frame_record(i, i / 30, fighter if i % 2 == 0 else None, [fighter]) for i in range(10)]
    meta = {"fps": 30.0, "width": 2560, "height": 1440, "frame_count": 10, "opponent_id": None}
    path = tmp_path / "fighter_1.json"
    write_keypoints_json(path, meta, frames)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert {k: data[k] for k in meta} == meta
    assert len(data["frames"]) == 10
    assert [f["frame"] for f in data["frames"]] == list(range(10))
    assert [f["found"] for f in data["frames"]] == [i % 2 == 0 for i in range(10)]
