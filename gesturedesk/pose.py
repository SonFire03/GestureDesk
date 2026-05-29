from __future__ import annotations

from typing import Any


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
