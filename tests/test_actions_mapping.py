import time

from gesturedesk.actions import CooldownManager, map_gesture_to_action
from gesturedesk.actions import ActionExecutor


def test_cooldown_blocks_repeated_action():
    cd = CooldownManager(cooldown_seconds=0.2)
    assert cd.ready("left_click") is True
    assert cd.ready("left_click") is False
    time.sleep(0.21)
    assert cd.ready("left_click") is True


def test_mapping_disarmed_is_safe():
    action = map_gesture_to_action(
        gesture="pinch",
        armed=False,
        enable_mouse_control=True,
        enable_media_keys=True,
        enable_scroll=True,
        fast_open_palm=True,
    )
    assert action == "none"


def test_mapping_armed_click():
    action = map_gesture_to_action(
        gesture="pinch",
        armed=True,
        enable_mouse_control=True,
        enable_media_keys=True,
        enable_scroll=True,
        fast_open_palm=False,
    )
    assert action == "left_click"


def test_mapping_armed_two_fingers_drag_toggle():
    action = map_gesture_to_action(
        gesture="two_fingers",
        armed=True,
        enable_mouse_control=True,
        enable_media_keys=True,
        enable_scroll=True,
        fast_open_palm=False,
    )
    assert action == "toggle_drag"


def test_mapping_armed_fist_drag_toggle():
    action = map_gesture_to_action(
        gesture="fist",
        armed=True,
        enable_mouse_control=True,
        enable_media_keys=True,
        enable_scroll=True,
        fast_open_palm=False,
    )
    assert action == "toggle_drag"




def test_adaptive_gain_increases_with_motion():
    executor = ActionExecutor(
        cooldown_seconds=0.4,
        click_cooldown_seconds=0.35,
        mouse_smoothing_alpha=0.45,
        mouse_adaptive_gain_min=0.7,
        mouse_adaptive_gain_max=2.0,
        mouse_adaptive_scale_px=220.0,
        scroll_step=90,
        logger=type("L", (), {"warning": lambda *args, **kwargs: None})(),
    )
    small = executor._adaptive_gain(10.0)
    large = executor._adaptive_gain(200.0)
    assert large > small
