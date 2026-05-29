from gesturedesk.pose import recognize_body_gesture


class LM:
    def __init__(self, x: float, y: float, visibility: float = 1.0) -> None:
        self.x = x
        self.y = y
        self.visibility = visibility


def _mk_pose(ls_y=0.6, rs_y=0.6, lw_y=0.8, rw_y=0.8):
    pts = [LM(0.5, 0.5) for _ in range(33)]
    pts[11] = LM(0.4, ls_y, 1.0)  # left shoulder
    pts[12] = LM(0.6, rs_y, 1.0)  # right shoulder
    pts[15] = LM(0.35, lw_y, 1.0)  # left wrist
    pts[16] = LM(0.65, rw_y, 1.0)  # right wrist
    return pts


def test_pose_right_hand_up():
    pose = _mk_pose(lw_y=0.8, rw_y=0.4)
    assert recognize_body_gesture(pose) == "right_hand_up"


def test_pose_left_hand_up():
    pose = _mk_pose(lw_y=0.4, rw_y=0.8)
    assert recognize_body_gesture(pose) == "left_hand_up"


def test_pose_both_hands_up():
    pose = _mk_pose(lw_y=0.4, rw_y=0.4)
    assert recognize_body_gesture(pose) == "both_hands_up"
