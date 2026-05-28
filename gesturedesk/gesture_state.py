from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass


@dataclass
class GestureDecision:
    stable_gesture: str
    action: str


class GestureStateMachine:
    """Temporal gesture stabilizer + drag-toggle guard."""

    def __init__(self, history_size: int = 5, drag_hold_seconds: float = 0.22) -> None:
        self.history: deque[str] = deque(maxlen=max(3, history_size))
        self.drag_hold_seconds = drag_hold_seconds
        self._two_fingers_started_at: float | None = None
        self._two_fingers_latched = False

    def _majority(self) -> str:
        if not self.history:
            return "idle"
        counts = Counter(self.history)
        gesture, count = counts.most_common(1)[0]
        if count >= (len(self.history) // 2 + 1):
            return gesture
        return self.history[-1]

    def step(self, raw_gesture: str, proposed_action: str, now: float) -> GestureDecision:
        self.history.append(raw_gesture)
        stable = self._majority()
        action = proposed_action

        if stable != "two_fingers":
            self._two_fingers_started_at = None
            self._two_fingers_latched = False
        elif action == "toggle_drag":
            if self._two_fingers_latched:
                action = "none"
            else:
                if self._two_fingers_started_at is None:
                    self._two_fingers_started_at = now
                    action = "none"
                elif now - self._two_fingers_started_at < self.drag_hold_seconds:
                    action = "none"
                else:
                    self._two_fingers_latched = True

        return GestureDecision(stable_gesture=stable, action=action)
