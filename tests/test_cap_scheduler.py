from affine.src.scorer.cap_scheduler import compute_caps


def test_empty_input_returns_empty():
    assert compute_caps({}) == {}


def test_equal_remaining_split_equally():
    caps = compute_caps(
        {"A": 100, "B": 100, "C": 100, "D": 100, "E": 100},
        global_budget=500, min_cap=5, max_cap_per_env=200,
    )
    assert caps == {"A": 100, "B": 100, "C": 100, "D": 100, "E": 100}


def test_dominant_env_gets_capped_at_max_cap_per_env():
    caps = compute_caps(
        {"BIG": 1000, "SMALL_A": 10, "SMALL_B": 10},
        global_budget=480, min_cap=5, max_cap_per_env=200,
    )
    assert caps["BIG"] == 200  # clamped, not 470
    assert caps["SMALL_A"] == 5  # int(480*10/1020)=4 → floored to 5
    assert caps["SMALL_B"] == 5


def test_done_env_gets_min_cap_and_doesnt_steal_budget():
    caps = compute_caps(
        {"DONE": 0, "ACTIVE": 100},
        global_budget=480, min_cap=5, max_cap_per_env=200,
    )
    assert caps["DONE"] == 5  # exactly min_cap
    # ACTIVE got full share of remaining=100, but clamped to max_cap_per_env
    assert caps["ACTIVE"] == 200


def test_negative_remaining_treated_as_done():
    caps = compute_caps({"OVERSHOT": -5, "OK": 50}, global_budget=480)
    assert caps["OVERSHOT"] == 5


def test_all_done_returns_min_cap_for_all():
    caps = compute_caps({"A": 0, "B": 0, "C": 0}, global_budget=480, min_cap=5)
    assert caps == {"A": 5, "B": 5, "C": 5}


def test_proportional_allocation_when_no_clamps_apply():
    # Budget 100, remaining ratios 3:1:1, no env hits min/max bounds
    caps = compute_caps(
        {"A": 300, "B": 100, "C": 100},
        global_budget=100, min_cap=1, max_cap_per_env=1000,
    )
    # int(100*300/500)=60, int(100*100/500)=20, int(100*100/500)=20
    assert caps == {"A": 60, "B": 20, "C": 20}


def test_min_cap_floor_applies_to_rounding_to_zero():
    # One huge env starves a tiny one in raw math, but min_cap kicks in
    caps = compute_caps(
        {"HUGE": 1_000_000, "TINY": 1},
        global_budget=480, min_cap=5, max_cap_per_env=200,
    )
    assert caps["TINY"] == 5  # int(480*1/1000001)=0 → floored


def test_returns_only_input_envs():
    caps = compute_caps({"X": 50, "Y": 50}, global_budget=200)
    assert set(caps.keys()) == {"X", "Y"}
