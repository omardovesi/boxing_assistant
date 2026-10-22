from pipeline import config
from pipeline.labelling_kit import Candidate, select_guard_frames


def clip(seconds: int = 120, fps: int = 10) -> list[Candidate]:
    """Two fighters over a clip, guard heights cycling through low, mid and
    high so every category exists all along the clip."""
    out = []
    for n in range(seconds * fps):
        t = n / fps
        for k, fighter in enumerate(("fighter_1", "fighter_2")):
            left = ((n + 7 * k) % 50) / 50 - 0.8  # -0.8 .. +0.18
            right = ((n + 13 + 7 * k) % 50) / 50 - 0.8
            out.append(Candidate(n, t, fighter, {"left": left, "right": right}, weight=1.0))
    return out


def test_picks_n_frames_with_about_a_third_borderline():
    picks = select_guard_frames(clip(), n=20)
    assert len(picks) == 20
    cats = [p.category for p in picks]
    assert cats.count("borderline") == 7
    assert cats.count("up") == 7 and cats.count("down") == 6


def test_categories_match_guard_heights():
    picks = select_guard_frames(clip(), n=20)
    g = {cat: [p.candidate.guard[p.hand] for p in picks if p.category == cat] for cat in ("up", "down", "borderline")}
    assert max(g["down"]) < min(g["borderline"]) <= max(g["borderline"]) < min(g["up"])


def test_covers_both_fighters_and_both_hands():
    picks = select_guard_frames(clip(), n=20)
    assert {p.candidate.fighter for p in picks} == {"fighter_1", "fighter_2"}
    assert {p.hand for p in picks} == {"left", "right"}
    assert {(p.candidate.fighter, p.hand) for p in picks} == {
        (f, h) for f in ("fighter_1", "fighter_2") for h in ("left", "right")}


def test_spread_over_the_clip_and_apart_in_time():
    picks = select_guard_frames(clip(seconds=120), n=20, min_gap=1.5)
    times = [p.candidate.time for p in picks]
    assert times == sorted(times)
    assert min(times) < 12 and max(times) > 108  # first and last tenth of the clip
    assert len({int(t // 6) for t in times}) >= 18  # nearly every 6 s bin has one
    assert all(b - a >= 1.5 for a, b in zip(times, times[1:]))


def test_skips_untrusted_and_repeated_frames():
    cands = clip()
    for i, c in enumerate(cands):
        if i % 3 == 0:
            c.weight = 0.0
        elif i % 3 == 1:
            c.weight = config.MIN_EVENT_WEIGHT - 0.01
        else:
            c.repeat = i % 2 == 0
    picks = select_guard_frames(cands, n=20)
    assert picks
    assert all(p.candidate.weight >= config.MIN_EVENT_WEIGHT and not p.candidate.repeat for p in picks)


def test_hands_without_a_guard_value_are_not_picked():
    cands = [Candidate(n, n / 10, "fighter_1", {"left": None, "right": (n % 50) / 50}, 1.0) for n in range(600)]
    picks = select_guard_frames(cands, n=10)
    assert picks and all(p.hand == "right" for p in picks)


def test_nothing_usable_gives_no_picks():
    assert select_guard_frames([Candidate(0, 0.0, "fighter_1", {"left": 0.1, "right": 0.1}, 0.0)]) == []
