from types import SimpleNamespace

from gesturedesk.hand_selection import select_dominant_hand_index


def _hand(cx, cy, sx, sy):
    out = []
    for i in range(21):
        x = cx + ((i % 5) / 4.0 - 0.5) * sx
        y = cy + ((i // 5) / 4.0 - 0.5) * sy
        out.append(SimpleNamespace(x=x, y=y))
    return out


def test_select_auto_largest_bbox():
    small = _hand(0.3, 0.5, 0.08, 0.10)
    large = _hand(0.6, 0.5, 0.20, 0.22)
    assert select_dominant_hand_index([small, large], mode="auto") == 1


def test_select_left_prefers_high_x_in_mirrored_view():
    left_visual = _hand(0.8, 0.5, 0.12, 0.12)
    right_visual = _hand(0.2, 0.5, 0.12, 0.12)
    assert select_dominant_hand_index([right_visual, left_visual], mode="left") == 1


def test_select_right_prefers_low_x_in_mirrored_view():
    left_visual = _hand(0.8, 0.5, 0.12, 0.12)
    right_visual = _hand(0.2, 0.5, 0.12, 0.12)
    assert select_dominant_hand_index([right_visual, left_visual], mode="right") == 0
