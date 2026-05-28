from gesturedesk.runtime_utils import is_point_in_active_zone, majority_vote_gesture


def test_majority_vote_returns_majority():
    assert majority_vote_gesture(["index", "index", "idle"]) == "index"


def test_majority_vote_fallback_to_last_if_no_majority():
    assert majority_vote_gesture(["index", "idle", "two_fingers", "idle"]) == "idle"


def test_active_zone_true():
    assert is_point_in_active_zone((0.5, 0.5), 0.1) is True


def test_active_zone_false_near_border():
    assert is_point_in_active_zone((0.03, 0.5), 0.08) is False
