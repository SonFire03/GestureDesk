from gesturedesk.gestures import count_raised_fingers, distance, recognize_gesture


def test_distance_simple():
    assert distance((0.0, 0.0), (3.0, 4.0)) == 5.0


def test_count_raised_fingers_simulated_two():
    points = {
        3: (0.2, 0.5),
        4: (0.3, 0.5),  # thumb raised (x heuristic)
        6: (0.5, 0.6),
        8: (0.5, 0.4),  # index up
        10: (0.6, 0.6),
        12: (0.6, 0.7),  # middle down
        14: (0.7, 0.6),
        16: (0.7, 0.7),  # ring down
        18: (0.8, 0.6),
        20: (0.8, 0.7),  # pinky down
    }
    assert count_raised_fingers(points) == 2


def test_recognize_index_when_only_index_is_up_even_if_thumb_noisy():
    points = {
        3: (0.2, 0.5),
        4: (0.35, 0.5),  # thumb may appear "up" with x heuristic
        6: (0.5, 0.6),
        8: (0.5, 0.4),   # index up
        10: (0.6, 0.6),
        12: (0.6, 0.7),  # middle down
        14: (0.7, 0.6),
        16: (0.7, 0.7),  # ring down
        18: (0.8, 0.6),
        20: (0.8, 0.7),  # pinky down
    }
    assert recognize_gesture(points) == "index"
