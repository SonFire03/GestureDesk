from __future__ import annotations

import time
from dataclasses import dataclass, field

SAFE_ACTIONS = {
    "none",
    "move_mouse",
    "left_click",
    "scroll",
    "media_play_pause",
    "toggle_drag",
    "toggle_armed",
    "force_disarmed",
}


@dataclass
class SafetyController:
    armed: bool = False
    _open_palm_started_at: float | None = None
    _last_action: str = "none"
    _open_palm_latched: bool = False

    def toggle_armed(self) -> bool:
        self.armed = not self.armed
        self._last_action = "toggle_armed"
        return self.armed

    def force_disarmed(self) -> None:
        self.armed = False
        self._last_action = "force_disarmed"

    def register_open_palm_hold(self, is_open_palm: bool, hold_seconds: float) -> bool:
        now = time.monotonic()
        if not is_open_palm:
            self._open_palm_started_at = None
            self._open_palm_latched = False
            return False

        # After one successful hold-toggle, require palm release before next toggle.
        if self._open_palm_latched:
            return False

        if self._open_palm_started_at is None:
            self._open_palm_started_at = now
            return False

        if now - self._open_palm_started_at >= hold_seconds:
            self._open_palm_started_at = None
            self._open_palm_latched = True
            self.toggle_armed()
            return True
        return False

    def can_execute(self, action: str) -> bool:
        if action not in SAFE_ACTIONS:
            return False
        if action in {"none", "toggle_armed", "force_disarmed"}:
            return True
        return self.armed
