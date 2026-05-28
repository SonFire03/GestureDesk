from __future__ import annotations

from collections import Counter


def majority_vote_gesture(history: list[str], fallback: str = "idle") -> str:
    if not history:
        return fallback
    counts = Counter(history)
    gesture, count = counts.most_common(1)[0]
    if count >= (len(history) // 2 + 1):
        return gesture
    return history[-1]


def is_point_in_active_zone(
    point: tuple[float, float] | None,
    margin: float,
) -> bool:
    if point is None:
        return False
    x, y = point
    m = max(0.0, min(0.45, margin))
    return m <= x <= (1.0 - m) and m <= y <= (1.0 - m)
