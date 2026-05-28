from gesturedesk.gesture_state import GestureStateMachine


def test_majority_stabilizes_gesture():
    sm = GestureStateMachine(history_size=5, drag_hold_seconds=0.2)
    out = None
    now = 0.0
    for g in ["idle", "index", "index", "index"]:
        out = sm.step(raw_gesture=g, proposed_action="none", now=now)
        now += 0.05
    assert out is not None
    assert out.stable_gesture == "index"


def test_toggle_drag_requires_hold_and_release():
    sm = GestureStateMachine(history_size=3, drag_hold_seconds=0.2)

    # first two_fingers frame: not ready
    d1 = sm.step("two_fingers", "toggle_drag", now=0.0)
    assert d1.action == "none"

    # before hold threshold: still blocked
    d2 = sm.step("two_fingers", "toggle_drag", now=0.1)
    assert d2.action == "none"

    # after hold threshold: toggle allowed once
    d3 = sm.step("two_fingers", "toggle_drag", now=0.25)
    assert d3.action == "toggle_drag"

    # while still in two_fingers: latched block
    d4 = sm.step("two_fingers", "toggle_drag", now=0.5)
    assert d4.action == "none"

    # release gesture resets latch
    _ = sm.step("idle", "none", now=0.6)
    d5 = sm.step("two_fingers", "toggle_drag", now=0.9)
    assert d5.action == "none"
