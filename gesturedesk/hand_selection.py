from __future__ import annotations


def hand_bbox_area(hand_landmarks) -> float:
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    return max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))


def hand_center_x(hand_landmarks) -> float:
    xs = [lm.x for lm in hand_landmarks]
    return sum(xs) / len(xs)


def select_dominant_hand_index(hands_landmarks, mode: str = "auto") -> int:
    if not hands_landmarks:
        return 0

    mode = (mode or "auto").lower().strip()
    if mode == "left":
        # mirrored camera view: visually left hand tends to higher x
        return max(range(len(hands_landmarks)), key=lambda i: hand_center_x(hands_landmarks[i]))
    if mode == "right":
        return min(range(len(hands_landmarks)), key=lambda i: hand_center_x(hands_landmarks[i]))

    areas = [hand_bbox_area(h) for h in hands_landmarks]
    return max(range(len(areas)), key=lambda i: areas[i])
