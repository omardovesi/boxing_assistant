import pytest

from pipeline import config
from pipeline.metrics import (
    body_confidence, body_confidence_spread, confidence_factor, count_events, frame_weight, overlap_factor, ramp,
    weight_summary, weighted_mean,
)


def keypoints(face_conf: float, body_conf: float) -> list[list[float]]:
    """17 [x, y, conf] keypoints: 5 face points, then 12 body points."""
    return [[0.5, 0.5, face_conf]] * 5 + [[0.5, 0.5, body_conf]] * 12


def found(overlap: float = 0.0, body_conf: float = 0.9) -> dict:
    return {"found": True, "overlap": overlap, "keypoints": keypoints(0.1, body_conf)}


MISSING = {"found": False}


def test_ramp_rising_and_clamped():
    assert ramp(0.2, 0.3, 0.7) == 0.0
    assert ramp(0.5, 0.3, 0.7) == pytest.approx(0.5)
    assert ramp(0.9, 0.3, 0.7) == 1.0


def test_ramp_falling():
    assert ramp(0.05, 0.3, 0.1) == 1.0
    assert ramp(0.2, 0.3, 0.1) == pytest.approx(0.5)
    assert ramp(0.4, 0.3, 0.1) == 0.0


def test_overlap_factor_at_limits():
    assert overlap_factor(0.0) == 1.0
    assert overlap_factor(config.WEIGHT_OVERLAP_FULL) == 1.0
    assert overlap_factor(0.20) == pytest.approx(0.5)
    assert overlap_factor(config.WEIGHT_OVERLAP_ZERO) == 0.0
    assert overlap_factor(0.6) == 0.0


def test_confidence_factor_at_limits():
    assert confidence_factor(0.1) == 0.0
    assert confidence_factor(config.WEIGHT_CONF_ZERO) == 0.0
    mid = (config.WEIGHT_CONF_ZERO + config.WEIGHT_CONF_FULL) / 2
    assert confidence_factor(mid) == pytest.approx(0.5)
    assert confidence_factor(config.WEIGHT_CONF_FULL) == pytest.approx(1.0)
    assert confidence_factor(0.95) == 1.0


def test_body_confidence_ignores_face_points():
    assert body_confidence(keypoints(face_conf=0.0, body_conf=0.8)) == pytest.approx(0.8)


def test_frame_weight_missing_is_zero():
    assert frame_weight(MISSING) == 0.0


def test_frame_weight_clean_confident_frame_is_one():
    assert frame_weight(found()) == 1.0


def test_frame_weight_multiplies_factors():
    # overlap 0.2 -> 0.5; body confidence at the ramp's midpoint -> 0.5
    mid = (config.WEIGHT_CONF_ZERO + config.WEIGHT_CONF_FULL) / 2
    assert frame_weight(found(overlap=0.2, body_conf=mid)) == pytest.approx(0.25)


def test_frame_weight_low_face_confidence_does_not_matter():
    # e.g. headgear hiding the face on an otherwise clear frame
    assert frame_weight({"found": True, "overlap": 0.0, "keypoints": keypoints(0.0, 0.9)}) == 1.0


def test_weighted_mean_weights_values():
    r = weighted_mean([10.0, 20.0], [1.0, 0.25])
    assert r.value == pytest.approx((10 + 5) / 1.25)
    assert r.coverage == pytest.approx(0.625)


def test_weighted_mean_skips_none_but_keeps_window_length():
    r = weighted_mean([10.0, None, 30.0, None], [1.0, 1.0, 1.0, 0.0])
    assert r.value == pytest.approx(20.0)
    assert r.coverage == pytest.approx(0.5)  # 2 of 4 frames count


def test_weighted_mean_no_evidence():
    r = weighted_mean([1.0, 2.0], [0.0, 0.0])
    assert r.value is None and r.coverage == 0.0


def test_weighted_mean_empty_window():
    r = weighted_mean([], [])
    assert r.value is None and r.coverage == 0.0


def test_count_events_ignores_light_frames_and_counts_threshold():
    flags = [True, True, True, False]
    weights = [0.49, config.MIN_EVENT_WEIGHT, 1.0, 1.0]
    r = count_events(flags, weights)
    assert r.value == 2
    assert r.coverage == pytest.approx(sum(weights) / 4)


def test_weight_summary_shares():
    frames = [found(), found(), found(overlap=0.15), MISSING]
    s = weight_summary(frames)
    assert (s["zero"], s["between"], s["one"]) == (0.25, 0.25, 0.5)
    assert s["zero"] + s["between"] + s["one"] == pytest.approx(1.0)
    assert s["event_eligible"] == 0.75  # weights 1, 1, 0.75, 0
    assert s["coverage"] == pytest.approx(2.75 / 4)


def test_body_confidence_spread_uses_only_clean_found_frames():
    frames = [found(body_conf=c) for c in (0.5, 0.6, 0.7, 0.8, 0.9)]
    frames += [found(overlap=0.5, body_conf=0.1), MISSING]  # excluded
    s = body_confidence_spread(frames)
    assert s["frames"] == 5
    assert s["median"] == pytest.approx(0.7)
    assert s["p10"] < s["median"] < s["p90"]


def test_body_confidence_spread_too_few_frames():
    assert body_confidence_spread([found(), MISSING]) is None
