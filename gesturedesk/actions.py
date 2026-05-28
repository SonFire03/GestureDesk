from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CooldownManager:
    cooldown_seconds: float
    _last_call: dict[str, float] = field(default_factory=dict)

    def ready(self, action: str) -> bool:
        now = time.monotonic()
        previous = self._last_call.get(action)
        if previous is None or now - previous >= self.cooldown_seconds:
            self._last_call[action] = now
            return True
        return False


def map_gesture_to_action(
    gesture: str,
    armed: bool,
    enable_mouse_control: bool,
    enable_media_keys: bool,
    enable_scroll: bool,
    fast_open_palm: bool,
) -> str:
    if not armed:
        return "none"

    if gesture == "fist" and enable_mouse_control:
        return "toggle_drag"
    if gesture == "index" and enable_mouse_control:
        return "move_mouse"
    if gesture == "pinch" and enable_mouse_control:
        return "left_click"
    if gesture == "two_fingers" and enable_mouse_control:
        return "toggle_drag"
    if fast_open_palm and enable_media_keys:
        return "media_play_pause"
    return "none"


class ActionExecutor:
    def __init__(
        self,
        cooldown_seconds: float,
        click_cooldown_seconds: float,
        mouse_smoothing_alpha: float,
        mouse_adaptive_gain_min: float,
        mouse_adaptive_gain_max: float,
        mouse_adaptive_scale_px: float,
        scroll_step: int,
        logger: Any,
    ) -> None:
        self.cooldown = CooldownManager(cooldown_seconds=cooldown_seconds)
        self.click_cooldown = CooldownManager(cooldown_seconds=click_cooldown_seconds)
        self.mouse_smoothing_alpha = max(0.05, min(0.95, mouse_smoothing_alpha))
        self.mouse_adaptive_gain_min = max(0.1, mouse_adaptive_gain_min)
        self.mouse_adaptive_gain_max = max(self.mouse_adaptive_gain_min, mouse_adaptive_gain_max)
        self.mouse_adaptive_scale_px = max(1.0, mouse_adaptive_scale_px)
        self.scroll_step = int(scroll_step)
        self.logger = logger
        self._smoothed_mouse: tuple[float, float] | None = None
        self.drag_active = False
        self.available = True
        try:
            import pyautogui  # type: ignore

            pyautogui.FAILSAFE = True
            # Critical for realtime control: default PyAutoGUI pause (~0.1s) kills FPS.
            pyautogui.PAUSE = 0
            self.pyautogui = pyautogui
        except Exception as exc:  # pragma: no cover - environment dependent
            self.available = False
            self.pyautogui = None
            self.logger.warning("PyAutoGUI indisponible: %s", exc)

    def execute(self, action: str, index_point: tuple[float, float] | None = None) -> str:
        if action == "none":
            return "none"
        if not self.available:
            return "pyautogui_unavailable"

        if action == "move_mouse" and index_point is not None:
            screen_w, screen_h = self.pyautogui.size()
            target_x = float(index_point[0] * screen_w)
            target_y = float(index_point[1] * screen_h)
            x, y = self._smooth_mouse(target_x, target_y)
            self.pyautogui.moveTo(x, y, duration=0)
            return "mouse_moved"

        if action == "left_click" and self.click_cooldown.ready("left_click"):
            self.pyautogui.click()
            return "left_click"

        if action == "toggle_drag" and self.cooldown.ready("toggle_drag"):
            if not self.drag_active:
                self.pyautogui.mouseDown(button="left")
                self.drag_active = True
                return "drag_started"
            self.pyautogui.mouseUp(button="left")
            self.drag_active = False
            return "drag_stopped"

        if action == "scroll" and self.cooldown.ready("scroll"):
            self.pyautogui.scroll(-self.scroll_step)
            return "scroll"

        if action == "media_play_pause" and self.cooldown.ready("media_play_pause"):
            self.pyautogui.press("playpause")
            return "media_play_pause"


        return "cooldown_or_ignored"

    def _smooth_mouse(self, target_x: float, target_y: float) -> tuple[int, int]:
        if self._smoothed_mouse is None:
            self._smoothed_mouse = (target_x, target_y)
        else:
            px, py = self._smoothed_mouse
            delta = math.hypot(target_x - px, target_y - py)
            gain = self._adaptive_gain(delta)
            a = min(0.98, self.mouse_smoothing_alpha * gain)
            self._smoothed_mouse = ((1 - a) * px + a * target_x, (1 - a) * py + a * target_y)
        sx, sy = self._smoothed_mouse
        return int(sx), int(sy)

    def _adaptive_gain(self, delta_px: float) -> float:
        ratio = min(1.0, max(0.0, delta_px / self.mouse_adaptive_scale_px))
        return self.mouse_adaptive_gain_min + (
            self.mouse_adaptive_gain_max - self.mouse_adaptive_gain_min
        ) * ratio

    def release_all(self) -> None:
        if not self.available:
            return
        if self.drag_active:
            self.pyautogui.mouseUp(button="left")
            self.drag_active = False
