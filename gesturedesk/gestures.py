from __future__ import annotations

import math
from typing import Any


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def count_raised_fingers(points: dict[int, tuple[float, float]]) -> int:
    # Finger is considered raised when tip is above PIP joint (lower y in image coords).
    finger_pairs = [
        (8, 6),   # index
        (12, 10), # middle
        (16, 14), # ring
        (20, 18), # pinky
    ]
    raised = sum(1 for tip, pip in finger_pairs if points[tip][1] < points[pip][1])

    # Thumb simplified horizontal heuristic.
    if points[4][0] > points[3][0]:
        raised += 1
    return raised


def landmarks_to_points(hand_landmarks: Any) -> dict[int, tuple[float, float]]:
    if hasattr(hand_landmarks, "landmark"):
        iterable = hand_landmarks.landmark
    else:
        iterable = hand_landmarks
    return {idx: (lm.x, lm.y) for idx, lm in enumerate(iterable)}


def recognize_gesture(points: dict[int, tuple[float, float]]) -> str:
    fingers = count_raised_fingers(points)
    pinch = distance(points[4], points[8]) < 0.05

    if pinch:
        return "pinch"
    if fingers == 0:
        return "fist"
    if fingers >= 4:
        return "open_palm"

    index_up = points[8][1] < points[6][1]
    middle_up = points[12][1] < points[10][1]
    ring_up = points[16][1] < points[14][1]
    pinky_up = points[20][1] < points[18][1]

    # Index mode should stay reliable even if thumb heuristic is noisy.
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "index"
    if index_up and middle_up:
        return "two_fingers"

    return "idle"
