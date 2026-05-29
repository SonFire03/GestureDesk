from __future__ import annotations

from typing import Any

POSE_FULL_EDGES = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left leg / foot
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    # Right leg / foot
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

POSE_GROUP_COLORS = {
    "face": (170, 140, 255),
    "torso": (80, 220, 255),
    "left_arm": (60, 255, 170),
    "right_arm": (255, 190, 70),
    "left_leg": (80, 200, 255),
    "right_leg": (255, 120, 190),
}

POSE_GROUPS = {
    "face": {(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)},
    "torso": {(11, 12), (11, 23), (12, 24), (23, 24)},
    "left_arm": {(11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19)},
    "right_arm": {(12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)},
    "left_leg": {(23, 25), (25, 27), (27, 29), (29, 31), (27, 31)},
    "right_leg": {(24, 26), (26, 28), (28, 30), (30, 32), (28, 32)},
}


def _pt(landmarks: list[Any], idx: int):
    lm = landmarks[idx]
    return lm.x, lm.y, getattr(lm, "visibility", 1.0)


def recognize_body_gesture(
    pose_landmarks: list[Any] | None,
    visibility_threshold: float = 0.5,
) -> str:
    if not pose_landmarks or len(pose_landmarks) < 17:
        return "none"

    # MediaPipe Pose indices used:
    # left_shoulder=11, right_shoulder=12, left_wrist=15, right_wrist=16
    ls_x, ls_y, ls_v = _pt(pose_landmarks, 11)
    rs_x, rs_y, rs_v = _pt(pose_landmarks, 12)
    lw_x, lw_y, lw_v = _pt(pose_landmarks, 15)
    rw_x, rw_y, rw_v = _pt(pose_landmarks, 16)
    _ = (ls_x, rs_x, lw_x, rw_x)

    if min(ls_v, rs_v, lw_v, rw_v) < visibility_threshold:
        return "none"

    margin = 0.05
    left_up = lw_y < (ls_y - margin)
    right_up = rw_y < (rs_y - margin)

    if left_up and right_up:
        return "both_hands_up"
    if right_up:
        return "right_hand_up"
    if left_up:
        return "left_hand_up"
    return "none"
