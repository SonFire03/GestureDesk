from gesturedesk.safety import SafetyController


def test_toggle_armed_changes_state():
    safety = SafetyController(armed=False)
    assert safety.toggle_armed() is True
    assert safety.toggle_armed() is False


def test_force_disarmed():
    safety = SafetyController(armed=True)
    safety.force_disarmed()
    assert safety.armed is False


def test_can_execute_when_disarmed():
    safety = SafetyController(armed=False)
    assert safety.can_execute("none") is True
    assert safety.can_execute("left_click") is False


def test_open_palm_hold_requires_release_before_next_toggle():
    safety = SafetyController(armed=False)
    hold = 0.01

    # First hold toggles to armed.
    assert safety.register_open_palm_hold(is_open_palm=True, hold_seconds=hold) is False
    import time
    time.sleep(0.02)
    assert safety.register_open_palm_hold(is_open_palm=True, hold_seconds=hold) is True
    assert safety.armed is True

    # Keeping open palm should not retrigger without release.
    time.sleep(0.02)
    assert safety.register_open_palm_hold(is_open_palm=True, hold_seconds=hold) is False
    assert safety.armed is True

    # Release then hold again can toggle back.
    assert safety.register_open_palm_hold(is_open_palm=False, hold_seconds=hold) is False
    assert safety.register_open_palm_hold(is_open_palm=True, hold_seconds=hold) is False
    time.sleep(0.02)
    assert safety.register_open_palm_hold(is_open_palm=True, hold_seconds=hold) is True
    assert safety.armed is False
