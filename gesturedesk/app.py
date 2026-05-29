from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from gesturedesk.actions import ActionExecutor, map_gesture_to_action
from gesturedesk.camera import CameraError, CameraStream
from gesturedesk.config import AppConfig, save_config_updates
from gesturedesk.gesture_state import GestureStateMachine
from gesturedesk.gestures import landmarks_to_points, recognize_gesture
from gesturedesk.hand_selection import select_dominant_hand_index
from gesturedesk.runtime_utils import is_point_in_active_zone, majority_vote_gesture
from gesturedesk.safety import SafetyController
from gesturedesk.actions import map_body_gesture_to_action
from gesturedesk.pose import POSE_FULL_EDGES, POSE_GROUP_COLORS, POSE_GROUPS, recognize_body_gesture


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
FINGER_GROUPS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}
FINGER_COLORS = {
    "thumb": (255, 170, 70),
    "index": (60, 255, 170),
    "middle": (60, 230, 255),
    "ring": (220, 120, 255),
    "pinky": (255, 95, 175),
}
BG_ACCENT = (14, 10, 24)
GESTURE_ACCENT = {
    "idle": (160, 160, 185),
    "index": (60, 255, 170),
    "two_fingers": (60, 230, 255),
    "pinch": (255, 175, 90),
    "open_palm": (200, 120, 255),
    "fist": (150, 150, 170),
}
TIP_INDICES = {4, 8, 12, 16, 20}
THEMES = {
    "cyber": {
        "bg": (14, 10, 24),
        "hud_border": (95, 210, 255),
        "hud_text": (230, 245, 255),
        "muted": (190, 190, 190),
        "focus": (120, 255, 220),
    },
    "clean": {
        "bg": (28, 28, 28),
        "hud_border": (175, 175, 175),
        "hud_text": (245, 245, 245),
        "muted": (205, 205, 205),
        "focus": (180, 220, 255),
    },
    "minimal": {
        "bg": (10, 10, 10),
        "hud_border": (90, 90, 90),
        "hud_text": (220, 220, 220),
        "muted": (165, 165, 165),
        "focus": (150, 210, 255),
    },
}
CALIBRATION_TARGETS = [
    (0.50, 0.50, "center"),
    (0.08, 0.12, "top-left"),
    (0.92, 0.12, "top-right"),
    (0.92, 0.88, "bottom-right"),
    (0.08, 0.88, "bottom-left"),
]


def _create_hand_landmarker(config: AppConfig):
    model_path = Path(config.model_path)
    if not model_path.exists():
        raise RuntimeError(
            f"Modele MediaPipe introuvable: {model_path}. "
            "Ajoute un fichier local 'hand_landmarker.task' puis relance."
        )

    try:
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
        from mediapipe.tasks.python.vision.hand_landmarker import (
            HandLandmarker,
            HandLandmarkerOptions,
        )
    except Exception as exc:
        raise RuntimeError(
            f"MediaPipe Tasks indisponible dans cet environnement: {exc}"
        ) from exc

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
    return HandLandmarker.create_from_options(options)


def _create_pose_landmarker(config: AppConfig, model_path: str | None = None):
    model_path = Path(model_path or config.pose_model_path)
    if not model_path.exists():
        raise RuntimeError(
            f"Modele Pose Landmarker introuvable: {model_path}. "
            "Ajoute un fichier local 'pose_landmarker_lite.task' puis relance."
        )
    try:
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
        from mediapipe.tasks.python.vision.pose_landmarker import (
            PoseLandmarker,
            PoseLandmarkerOptions,
        )
    except Exception as exc:
        raise RuntimeError(f"MediaPipe Pose Tasks indisponible: {exc}") from exc

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=config.min_detection_confidence,
        min_pose_presence_confidence=config.min_tracking_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
    return PoseLandmarker.create_from_options(options)


class GestureDeskApp:
    def __init__(self, config: AppConfig, logger, config_path: str = "config.json") -> None:
        self.config = config
        self.logger = logger
        self.config_path = config_path
        self.safety = SafetyController(armed=False)
        self.actions = ActionExecutor(
            cooldown_seconds=config.cooldown_seconds,
            click_cooldown_seconds=config.click_cooldown_seconds,
            mouse_smoothing_alpha=config.mouse_smoothing_alpha,
            mouse_adaptive_gain_min=config.mouse_adaptive_gain_min,
            mouse_adaptive_gain_max=config.mouse_adaptive_gain_max,
            mouse_adaptive_scale_px=config.mouse_adaptive_scale_px,
            scroll_step=config.scroll_step,
            logger=logger,
        )
        self._last_action_label = "none"
        self._last_open_palm_seen = 0.0
        self._two_fingers_started_at: float | None = None
        self._two_fingers_latched = False
        self._index_trail: deque[tuple[int, int]] = deque(maxlen=18)
        self._frame_times: deque[float] = deque(maxlen=30)
        self._frame_idx = 0
        self._last_detection_result = None
        self._inference_stride = max(1, int(self.config.inference_every_n_frames))
        self._inference_scale = float(self.config.inference_scale)
        self._pose_inference_stride = max(1, int(self.config.pose_inference_every_n_frames))
        self._pose_inference_scale = max(0.25, min(1.0, float(self.config.pose_inference_scale)))
        self._last_mp_timestamp_ms = -1
        self._gesture_history: deque[str] = deque(maxlen=5)
        self._active_zone_margin = max(0.01, min(0.25, config.active_zone_margin))
        self._profile_name = "balanced"
        self._state = GestureStateMachine(
            history_size=5,
            drag_hold_seconds=config.drag_toggle_hold_seconds,
            click_hold_seconds=config.pinch_click_hold_seconds,
        )
        self._calibrating = False
        self._calibration_started_at = 0.0
        self._calibration_samples: list[tuple[float, float]] = []
        self._calib_stage_index = 0
        self._calib_stage_started_at = 0.0
        self._calib_stage_seconds = max(0.6, float(config.calibration_stage_seconds))
        self._calib_min_x = max(0.0, min(1.0, config.calib_min_x))
        self._calib_max_x = max(self._calib_min_x + 1e-6, min(1.0, config.calib_max_x))
        self._calib_min_y = max(0.0, min(1.0, config.calib_min_y))
        self._calib_max_y = max(self._calib_min_y + 1e-6, min(1.0, config.calib_max_y))
        self._t_capture_ms = 0.0
        self._t_infer_ms = 0.0
        self._t_pose_ms = 0.0
        self._t_render_ms = 0.0
        self._theme_name = "cyber"
        self._gesture_confidence = 1.0
        self._gesture_timeline: deque[str] = deque(maxlen=14)
        self._body_both_started_at: float | None = None
        self._last_body_gesture = "none"
        self._pose_model_label = Path(self.config.pose_model_path).stem
        self._last_pose_landmarks = None
        self._ui_mode = config.ui_mode if config.ui_mode in {"pro", "debug"} else "pro"
        self._show_skeleton_window = True
        self._draw_path_enabled = bool(config.draw_path)
        self._hsv_fallback_enabled = bool(config.enable_hsv_fallback)
        self._studio_open = False
        self._studio_idx = 0
        self._studio_items = [
            ("inference_scale", "hand scale", 0.40, 0.95, 0.02),
            ("inference_every_n_frames", "hand stride", 1, 5, 1),
            ("pose_inference_scale", "pose scale", 0.30, 0.90, 0.02),
            ("pose_inference_every_n_frames", "pose stride", 1, 6, 1),
            ("mouse_smoothing_alpha", "mouse smooth", 0.10, 0.85, 0.02),
            ("active_zone_margin", "active margin", 0.00, 0.20, 0.01),
            ("click_cooldown_seconds", "click cooldown", 0.10, 0.80, 0.01),
            ("drag_toggle_hold_seconds", "drag hold", 0.10, 0.60, 0.01),
            ("pinch_click_hold_seconds", "pinch hold", 0.05, 0.40, 0.01),
        ]

    def run(self) -> int:
        try:
            cam = CameraStream(
                camera_id=self.config.camera_id,
                width=self.config.camera_width,
                height=self.config.camera_height,
                fps=self.config.camera_fps,
                fourcc=self.config.camera_fourcc,
                autofocus=self.config.camera_autofocus,
                auto_exposure=self.config.camera_auto_exposure,
                exposure=self.config.camera_exposure,
            )
        except CameraError as exc:
            self.logger.error(str(exc))
            print(str(exc))
            return 1

        try:
            hand_landmarker = _create_hand_landmarker(self.config)
        except RuntimeError as exc:
            self.logger.error(str(exc))
            print(str(exc))
            cam.release()
            return 1

        with hand_landmarker:
            pose_landmarker = None
            if self.config.enable_body_control:
                try:
                    pose_landmarker = _create_pose_landmarker(self.config)
                except Exception as exc:
                    pose_landmarker = None
                    self.logger.warning("Body control desactive (Pose indisponible): %s", exc)
            self.logger.info("Application demarree. Camera id=%s", self.config.camera_id)
            cv2.namedWindow("GestureDesk", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("GestureDesk", 1400, 900)
            if self._show_skeleton_window:
                cv2.namedWindow("GestureDesk Skeleton", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("GestureDesk Skeleton", 960, 540)
            while True:
                t_loop = time.monotonic()
                t0 = time.monotonic()
                frame = cam.read()
                self._t_capture_ms = (time.monotonic() - t0) * 1000.0
                if frame is None:
                    self.logger.error("Lecture camera echouee")
                    break

                frame = cv2.flip(frame, 1)
                self._frame_idx += 1
                self._frame_times.append(time.monotonic())
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self._inference_scale < 0.99:
                    rgb_for_inference = cv2.resize(
                        rgb,
                        None,
                        fx=self._inference_scale,
                        fy=self._inference_scale,
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    rgb_for_inference = rgb
                timestamp_ms = int(time.monotonic() * 1000)
                if timestamp_ms <= self._last_mp_timestamp_ms:
                    timestamp_ms = self._last_mp_timestamp_ms + 1
                self._last_mp_timestamp_ms = timestamp_ms
                if self._frame_idx % self._inference_stride == 0 or self._last_detection_result is None:
                    t_inf = time.monotonic()
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_for_inference)
                    self._last_detection_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                    self._t_infer_ms = (time.monotonic() - t_inf) * 1000.0
                result = self._last_detection_result

                gesture = "idle"
                index_point = None
                fast_open_palm = False
                body_gesture = "none"
                pose_landmarks = None

                if result.hand_landmarks:
                    dominant_idx = select_dominant_hand_index(
                        result.hand_landmarks,
                        mode=self.config.dominant_hand_mode,
                    )
                    for idx, hand_landmarks in enumerate(result.hand_landmarks):
                        is_dominant = idx == dominant_idx
                        if (not is_dominant) and (not self.config.draw_secondary_hand):
                            continue
                        if not self.config.ui_minimal:
                            self._draw_hand_landmarks(frame, hand_landmarks, is_dominant=is_dominant)
                        points = landmarks_to_points(hand_landmarks)
                        hand_gesture = recognize_gesture(points)
                        if not self.config.ui_minimal:
                            label_suffix = "DOM" if is_dominant else "2ND"
                            self._draw_gesture_tag_near_hand(frame, hand_landmarks, f"{hand_gesture} [{label_suffix}]")
                        if is_dominant:
                            gesture = hand_gesture
                            index_point = points.get(8)
                            if not self.config.ui_minimal:
                                self._draw_dominant_hand_focus(frame, hand_landmarks, hand_gesture)
                            # Finger card removed on user request.
                else:
                    self._index_trail.clear()

                if index_point is None and self._hsv_fallback_enabled:
                    hsv_point = self._detect_hsv_fallback_point(frame)
                    if hsv_point is not None:
                        index_point = hsv_point
                        gesture = "index"
                        self._last_action_label = "hsv_fallback"

                if pose_landmarker is not None:
                    if self._pose_inference_scale < 0.99:
                        rgb_for_pose = cv2.resize(
                            rgb,
                            None,
                            fx=self._pose_inference_scale,
                            fy=self._pose_inference_scale,
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        rgb_for_pose = rgb
                    if self._frame_idx % self._pose_inference_stride == 0 or self._last_pose_landmarks is None:
                        t_pose = time.monotonic()
                        mp_pose_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_for_pose)
                        pose_result = pose_landmarker.detect_for_video(mp_pose_image, timestamp_ms)
                        self._last_pose_landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
                        self._t_pose_ms = (time.monotonic() - t_pose) * 1000.0
                    pose_landmarks = self._last_pose_landmarks
                    body_gesture = recognize_body_gesture(pose_landmarks)
                    self._last_body_gesture = body_gesture
                    if pose_landmarks and self.config.draw_pose_overlay and not self.config.ui_minimal:
                        self._draw_pose_overlay(frame, pose_landmarks)

                self._gesture_history.append(gesture)
                gesture = majority_vote_gesture(list(self._gesture_history), fallback=gesture)
                self._gesture_confidence = self._estimate_gesture_confidence(gesture)

                if self._calibrating:
                    self._update_calibration_wizard(index_point)

                now = time.monotonic()
                if gesture == "open_palm":
                    if now - self._last_open_palm_seen < 0.35:
                        fast_open_palm = True
                    self._last_open_palm_seen = now

                action = map_gesture_to_action(
                    gesture=gesture,
                    armed=self.safety.armed,
                    enable_mouse_control=self.config.enable_mouse_control,
                    enable_media_keys=self.config.enable_media_keys,
                    enable_scroll=self.config.enable_scroll,
                    fast_open_palm=fast_open_palm,
                )

                decision = self._state.step(raw_gesture=gesture, proposed_action=action, now=time.monotonic())
                gesture = decision.stable_gesture
                action = decision.action

                if body_gesture == "both_hands_up":
                    if self._body_both_started_at is None:
                        self._body_both_started_at = now
                    elif now - self._body_both_started_at >= self.config.body_hold_seconds:
                        body_gesture = "both_hands_up_hold"
                else:
                    self._body_both_started_at = None

                if action == "none":
                    action = map_body_gesture_to_action(
                        body_gesture=body_gesture,
                        armed=self.safety.armed,
                        enable_media_keys=self.config.enable_media_keys,
                    )

                if self.safety.can_execute(action):
                    mapped_point = index_point
                    if action == "move_mouse":
                        mapped_point = self._map_calibrated_point(index_point)
                        if not is_point_in_active_zone(mapped_point, self._active_zone_margin):
                            action = "none"
                    result_label = self.actions.execute(action, index_point=mapped_point)
                    if result_label != "none":
                        self._last_action_label = result_label

                t_rnd = time.monotonic()
                self._draw_overlay(
                    frame,
                    gesture,
                    hands_count=len(result.hand_landmarks) if result.hand_landmarks else 0,
                    fps=self._compute_fps(),
                )
                if self._calibrating:
                    self._draw_calibration_wizard(frame)
                self._draw_active_zone(frame)
                if self._studio_open:
                    self._draw_studio_panel(frame)
                self._t_render_ms = (time.monotonic() - t_rnd) * 1000.0
                cv2.imshow("GestureDesk", frame)

                # Secondary skeleton-only window (hands + pose) for clean tracking visualization.
                skeleton_frame = np.zeros_like(frame)
                if result.hand_landmarks:
                    dominant_idx = select_dominant_hand_index(
                        result.hand_landmarks,
                        mode=self.config.dominant_hand_mode,
                    )
                    for idx, hand_landmarks in enumerate(result.hand_landmarks):
                        is_dominant = idx == dominant_idx
                        if (not is_dominant) and (not self.config.draw_secondary_hand):
                            continue
                        self._draw_hand_landmarks(skeleton_frame, hand_landmarks, is_dominant=is_dominant)
                if pose_landmarks and self.config.draw_pose_overlay:
                    self._draw_pose_overlay(skeleton_frame, pose_landmarks)
                if self._show_skeleton_window:
                    cv2.imshow("GestureDesk Skeleton", skeleton_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.logger.info("Quit requested")
                    break
                if key == ord("a"):
                    new_state = self.safety.toggle_armed()
                    self._last_action_label = f"armed={new_state} (manual)"
                    self.logger.info("Toggle armed manual: %s", new_state)
                    if not new_state:
                        self.actions.release_all()
                if key == ord("d"):
                    self.safety.force_disarmed()
                    self.actions.release_all()
                    self._last_action_label = "forced_disarmed"
                    self.logger.info("Forced disarmed")
                if key == ord("c"):
                    self._start_calibration()
                if key == ord("1"):
                    self._apply_profile("precision")
                if key == ord("2"):
                    self._apply_profile("balanced")
                if key == ord("3"):
                    self._apply_profile("performance")
                if key == ord("t"):
                    self._cycle_theme()
                if key == ord("u"):
                    self._toggle_ui_mode()
                if key == ord("o"):
                    self._studio_open = not self._studio_open
                    self._last_action_label = f"studio={'on' if self._studio_open else 'off'}"
                if key == ord("p"):
                    self._draw_path_enabled = not self._draw_path_enabled
                    save_config_updates(self.config_path, {"draw_path": self._draw_path_enabled})
                    self._last_action_label = f"draw_path={self._draw_path_enabled}"
                if key == ord("f"):
                    self._hsv_fallback_enabled = not self._hsv_fallback_enabled
                    save_config_updates(self.config_path, {"enable_hsv_fallback": self._hsv_fallback_enabled})
                    self._last_action_label = f"hsv_fallback={self._hsv_fallback_enabled}"
                if key == ord("v"):
                    self._show_skeleton_window = not self._show_skeleton_window
                    if self._show_skeleton_window:
                        cv2.namedWindow("GestureDesk Skeleton", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("GestureDesk Skeleton", 960, 540)
                    else:
                        cv2.destroyWindow("GestureDesk Skeleton")
                    self._last_action_label = f"skeleton_win={self._show_skeleton_window}"
                if self._studio_open and key in (ord("j"), 84):
                    self._studio_idx = (self._studio_idx + 1) % len(self._studio_items)
                if self._studio_open and key in (ord("k"), 82):
                    self._studio_idx = (self._studio_idx - 1) % len(self._studio_items)
                if self._studio_open and key in (ord("h"), 81):
                    self._studio_adjust(-1)
                if self._studio_open and key in (ord("l"), 83):
                    self._studio_adjust(1)
                if key == ord("7"):
                    pose_landmarker = self._disable_body_control_runtime(pose_landmarker)
                if key == ord("8"):
                    pose_landmarker = self._switch_pose_model(
                        pose_landmarker, "models/pose_landmarker_lite.task"
                    )
                if key == ord("9"):
                    pose_landmarker = self._switch_pose_model(
                        pose_landmarker, "models/pose_landmarker_full.task"
                    )
                _ = t_loop

        self.actions.release_all()
        if "pose_landmarker" in locals() and pose_landmarker is not None:
            pose_landmarker.close()
        cam.release()
        cv2.destroyAllWindows()
        self.logger.info("Application stopped")
        return 0

    def _compute_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        duration = self._frame_times[-1] - self._frame_times[0]
        if duration <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / duration

    def _blend_roi(self, frame, x1: int, y1: int, x2: int, y2: int, alpha: float, color) -> None:
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return
        roi = frame[y1:y2, x1:x2]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), color, -1)
        frame[y1:y2, x1:x2] = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)

    def _draw_hand_landmarks(self, frame, hand_landmarks, is_dominant: bool) -> None:
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        # Palm fill for better hand shape perception.
        palm_idx = [0, 1, 5, 9, 13, 17]
        palm_poly = [pts[i] for i in palm_idx if i < len(pts)]
        if len(palm_poly) >= 3:
            hull = cv2.convexHull(np.array(palm_poly, dtype=np.int32))
            fill_color = (36, 62, 105) if is_dominant else (24, 28, 36)
            cv2.fillConvexPoly(frame, hull, fill_color)
            cv2.polylines(frame, [hull], True, (95, 215, 255) if is_dominant else (78, 82, 96), 1, cv2.LINE_AA)

        # Palm / skeleton base
        for start, end in HAND_CONNECTIONS:
            if start < len(pts) and end < len(pts):
                base_color = (100, 110, 145) if is_dominant else (64, 70, 84)
                cv2.line(frame, pts[start], pts[end], base_color, 2, cv2.LINE_AA)

        # Finger segments with color coding.
        for finger_name, joints in FINGER_GROUPS.items():
            color = FINGER_COLORS[finger_name]
            for idx in range(len(joints) - 1):
                s, e = joints[idx], joints[idx + 1]
                if s < len(pts) and e < len(pts):
                    thickness = 3 if is_dominant else 1
                    draw_color = color if is_dominant else tuple(int(c * 0.55) for c in color)
                    cv2.line(frame, pts[s], pts[e], draw_color, thickness, cv2.LINE_AA)
                    if is_dominant:
                        cv2.line(frame, pts[s], pts[e], draw_color, thickness + 1, cv2.LINE_AA)

        # Landmarks with tips emphasized.
        for idx, (x, y) in enumerate(pts):
            pulse = 1 if (self._frame_idx // 3) % 2 == 0 else 0
            radius = 6 + pulse if idx in TIP_INDICES else 3
            point_color = (255, 255, 255) if is_dominant else (180, 180, 180)
            cv2.circle(frame, (x, y), radius, point_color, -1, cv2.LINE_AA)
            if idx in TIP_INDICES and is_dominant:
                cv2.circle(frame, (x, y), radius + 2, (120, 255, 235), 1, cv2.LINE_AA)

        # Index fingertip halo for cursor targeting feedback.
        if 8 < len(pts) and is_dominant:
            x, y = pts[8]
            if self._draw_path_enabled:
                self._index_trail.append((x, y))
                # Trail every 2 frames to reduce GPU/CPU pressure.
                if self._frame_idx % 2 == 0:
                    self._draw_index_trail(frame)
            cv2.circle(frame, (x, y), 16, (120, 255, 215), 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 10, (70, 245, 180), 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 4, (150, 255, 230), -1, cv2.LINE_AA)

    def _draw_index_trail(self, frame) -> None:
        if len(self._index_trail) < 2:
            return
        pts = list(self._index_trail)
        for i in range(1, len(pts)):
            thickness = max(1, int(i / 4))
            alpha_color = (75, 120 + min(90, i * 3), 160 + min(80, i * 2))
            cv2.line(frame, pts[i - 1], pts[i], alpha_color, thickness, cv2.LINE_AA)

    def _draw_gesture_tag_near_hand(self, frame, hand_landmarks, gesture: str) -> None:
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        if not pts:
            return
        min_x = max(0, min(p[0] for p in pts) - 8)
        min_y = max(0, min(p[1] for p in pts) - 30)
        label = f"{gesture}"
        box_w = 155
        box_h = 28
        theme = THEMES[self._theme_name]
        self._blend_roi(frame, min_x, min_y - box_h, min_x + box_w, min_y + 4, 0.55, theme["bg"])
        cv2.rectangle(frame, (min_x, min_y - box_h), (min_x + box_w, min_y + 4), theme["hud_border"], 1)
        cv2.putText(frame, label, (min_x + 8, min_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, theme["hud_text"], 2, cv2.LINE_AA)

    def _draw_dominant_hand_focus(self, frame, hand_landmarks, gesture: str) -> None:
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        if not pts:
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = max(0, min(xs) - 10), min(w - 1, max(xs) + 10)
        y1, y2 = max(0, min(ys) - 10), min(h - 1, max(ys) + 10)
        accent = GESTURE_ACCENT.get(gesture, (180, 180, 180))
        # Thin, less intrusive focus box.
        cv2.rectangle(frame, (x1, y1), (x2, y2), accent, 1, cv2.LINE_AA)
        # Corner brackets for a more "HUD" visual.
        l = 9
        cv2.line(frame, (x1, y1), (x1 + l, y1), accent, 1, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + l), accent, 1, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2 - l, y1), accent, 1, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + l), accent, 1, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1 + l, y2), accent, 1, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - l), accent, 1, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2 - l, y2), accent, 1, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - l), accent, 1, cv2.LINE_AA)
        self._draw_gesture_stability_bar(frame, x1, y2 + 8, x2, gesture)

    def _draw_gesture_stability_bar(self, frame, x1: int, y: int, x2: int, gesture: str) -> None:
        bar_w = max(40, min(180, x2 - x1))
        bar_h = 8
        x = x1
        # Simple stability heuristic from trail availability.
        stability = min(1.0, len(self._index_trail) / 12.0) if gesture == "index" else 0.85
        self._blend_roi(frame, x, y, x + bar_w, y + bar_h, 0.45, (20, 20, 20))
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (95, 120, 140), 1)
        fill = int(bar_w * stability)
        cv2.rectangle(frame, (x + 1, y + 1), (x + fill, y + bar_h - 1), (80, 210, 255), -1)

    def _draw_finger_state_card(self, frame, points: dict[int, tuple[float, float]]) -> None:
        def is_up(tip: int, pip: int) -> bool:
            return points[tip][1] < points[pip][1]

        status = {
            "I": is_up(8, 6),
            "M": is_up(12, 10),
            "R": is_up(16, 14),
            "P": is_up(20, 18),
        }
        finger_card_colors = {
            "I": (0, 230, 145),
            "M": (0, 220, 255),
            "R": (255, 180, 0),
            "P": (255, 120, 120),
        }
        h, w = frame.shape[:2]
        card_w, card_h = 195, 82
        x0 = max(12, w - card_w - 12)
        y0 = 12
        self._blend_roi(frame, x0, y0, x0 + card_w, y0 + card_h, 0.5, BG_ACCENT)
        cv2.rectangle(frame, (x0, y0), (x0 + card_w, y0 + card_h), (95, 210, 255), 1)
        cv2.putText(frame, "Fingers", (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (232, 240, 255), 2, cv2.LINE_AA)
        labels = ["I", "M", "R", "P"]
        for idx, k in enumerate(labels):
            up = status[k]
            cx = x0 + 22 + idx * 42
            cy = y0 + 53
            color = finger_card_colors[k] if up else (95, 95, 100)
            cv2.circle(frame, (cx, cy), 11, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 11, (175, 235, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, k, (cx - 5, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_overlay(self, frame, gesture: str, hands_count: int, fps: float) -> None:
        theme = THEMES[self._theme_name]
        armed_label = "ARMED" if self.safety.armed else "DISARMED"
        state_color = (0, 255, 0) if self.safety.armed else (0, 0, 255)
        h, w = frame.shape[:2]
        x0, y0 = 12, 12
        panel_w, panel_h = (320, 98) if self._ui_mode == "pro" else (340, 120)
        fs = 0.46
        fs_small = 0.36
        lh = 18

        self._blend_roi(frame, x0, y0, x0 + panel_w, y0 + panel_h, 0.46, (18, 18, 18))
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.putText(frame, "GestureDesk", (x0 + 10, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, fs_small, (235, 235, 235), 1, cv2.LINE_AA)
        (tw, _th), _ = cv2.getTextSize(armed_label, cv2.FONT_HERSHEY_SIMPLEX, fs_small, 1)
        cv2.putText(frame, armed_label, (x0 + panel_w - tw - 10, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, fs_small, state_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{gesture} | {self._last_body_gesture}", (x0 + 10, y0 + 18 + lh), cv2.FONT_HERSHEY_SIMPLEX, fs, theme["hud_text"], 1, cv2.LINE_AA)
        cv2.putText(frame, f"Action: {self._last_action_label}", (x0 + 10, y0 + 18 + lh * 2), cv2.FONT_HERSHEY_SIMPLEX, fs_small, theme["hud_text"], 1, cv2.LINE_AA)
        cv2.putText(frame, f"{fps:.1f} fps   hands {hands_count}   {self._profile_name}", (x0 + 10, y0 + 18 + lh * 3), cv2.FONT_HERSHEY_SIMPLEX, fs_small, theme["hud_text"], 1, cv2.LINE_AA)
        cv2.putText(frame, f"conf {self._gesture_confidence:.2f}   ui {self._ui_mode}", (x0 + 10, y0 + 18 + lh * 4), cv2.FONT_HERSHEY_SIMPLEX, fs_small, theme["muted"], 1, cv2.LINE_AA)

        if self._ui_mode == "debug":
            keys = "q quit  a arm  d disarm  u ui  o studio  h/l adjust  p path  f hsv  v skel  7/8/9 body"
            self._blend_roi(frame, 12, h - 24, min(w - 12, 860), h - 8, 0.44, (18, 18, 18))
            cv2.rectangle(frame, (12, h - 24), (min(w - 12, 860), h - 8), (70, 70, 70), 1, cv2.LINE_AA)
            cv2.putText(frame, keys, (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, theme["hud_text"], 1, cv2.LINE_AA)

    def _draw_calibration_wizard(self, frame) -> None:
        h, w = frame.shape[:2]
        tx, ty, label = CALIBRATION_TARGETS[self._calib_stage_index]
        cx, cy = int(tx * w), int(ty * h)
        elapsed = time.monotonic() - self._calib_stage_started_at
        progress = max(0.0, min(1.0, elapsed / self._calib_stage_seconds))
        cv2.circle(frame, (cx, cy), 18, (90, 240, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)
        text = f"CALIB {self._calib_stage_index + 1}/{len(CALIBRATION_TARGETS)} {label}"
        cv2.putText(frame, text, (max(12, cx - 120), max(24, cy - 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 245, 255), 1, cv2.LINE_AA)
        bar_w, bar_h = 180, 10
        bx = max(10, min(w - bar_w - 10, cx - bar_w // 2))
        by = min(h - 16, cy + 24)
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (120, 140, 160), 1)
        cv2.rectangle(frame, (bx + 1, by + 1), (bx + int((bar_w - 2) * progress), by + bar_h - 1), (90, 230, 255), -1)

    def _draw_studio_panel(self, frame) -> None:
        theme = THEMES[self._theme_name]
        h, w = frame.shape[:2]
        panel_w = 360
        row_h = 20
        panel_h = 28 + row_h * (len(self._studio_items) + 1)
        x0 = max(10, w - panel_w - 10)
        y0 = max(60, h - panel_h - 10)
        self._blend_roi(frame, x0, y0, x0 + panel_w, y0 + panel_h, 0.40, theme["bg"])
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), theme["hud_border"], 1, cv2.LINE_AA)
        cv2.putText(frame, "STUDIO  o toggle | j/k select | h/l adjust", (x0 + 8, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.43, theme["hud_text"], 1, cv2.LINE_AA)
        for i, (_, label, *_rest) in enumerate(self._studio_items):
            y = y0 + 36 + i * row_h
            selected = i == self._studio_idx
            c = theme["focus"] if selected else theme["muted"]
            val = self._studio_value(self._studio_items[i][0])
            cv2.putText(frame, f"{label:16s} {val}", (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, c, 1, cv2.LINE_AA)

    def _studio_value(self, name: str) -> str:
        v = self._studio_get(name)
        if isinstance(v, int):
            return str(v)
        return f"{float(v):.2f}"

    def _studio_get(self, name: str):
        mapping = {
            "inference_scale": self._inference_scale,
            "inference_every_n_frames": self._inference_stride,
            "pose_inference_scale": self._pose_inference_scale,
            "pose_inference_every_n_frames": self._pose_inference_stride,
            "mouse_smoothing_alpha": self.actions.mouse_smoothing_alpha,
            "active_zone_margin": self._active_zone_margin,
            "click_cooldown_seconds": self.actions.click_cooldown.cooldown_seconds,
            "drag_toggle_hold_seconds": self._state.drag_hold_seconds,
            "pinch_click_hold_seconds": self._state.click_hold_seconds,
        }
        return mapping[name]

    def _studio_adjust(self, direction: int) -> None:
        name, _label, vmin, vmax, step = self._studio_items[self._studio_idx]
        old = self._studio_get(name)
        new = old + (step * direction)
        if isinstance(old, int):
            new = int(max(vmin, min(vmax, round(new))))
        else:
            new = float(max(vmin, min(vmax, new)))
        if new == old:
            return
        self._studio_set(name, new)
        save_config_updates(self.config_path, {name: new})
        self._last_action_label = f"{name}={new}"

    def _studio_set(self, name: str, value) -> None:
        if name == "inference_scale":
            self._inference_scale = float(value)
        elif name == "inference_every_n_frames":
            self._inference_stride = max(1, int(value))
        elif name == "pose_inference_scale":
            self._pose_inference_scale = float(value)
        elif name == "pose_inference_every_n_frames":
            self._pose_inference_stride = max(1, int(value))
        elif name == "mouse_smoothing_alpha":
            self.actions.mouse_smoothing_alpha = float(value)
        elif name == "active_zone_margin":
            self._active_zone_margin = float(value)
        elif name == "click_cooldown_seconds":
            self.actions.click_cooldown.cooldown_seconds = float(value)
        elif name == "drag_toggle_hold_seconds":
            self._state.drag_hold_seconds = float(value)
        elif name == "pinch_click_hold_seconds":
            self._state.click_hold_seconds = float(value)

    def _switch_pose_model(self, pose_landmarker, model_path: str):
        self.config = AppConfig(**{**self.config.__dict__, "enable_body_control": True, "pose_model_path": model_path})
        try:
            new_pose = _create_pose_landmarker(self.config, model_path=model_path)
            if pose_landmarker is not None:
                pose_landmarker.close()
            self._pose_model_label = Path(model_path).stem
            save_config_updates(
                self.config_path,
                {"pose_model_path": model_path, "enable_body_control": True},
            )
            self.logger.info("Pose model switched: %s", model_path)
            self._last_action_label = f"pose={self._pose_model_label}"
            return new_pose
        except Exception as exc:
            self.logger.warning("Pose model switch failed (%s): %s", model_path, exc)
            self._last_action_label = "pose_switch_failed"
            return pose_landmarker

    def _disable_body_control_runtime(self, pose_landmarker):
        if pose_landmarker is not None:
            pose_landmarker.close()
        self.config = AppConfig(**{**self.config.__dict__, "enable_body_control": False})
        self._pose_model_label = "hands_only"
        self._last_body_gesture = "none"
        self._body_both_started_at = None
        save_config_updates(self.config_path, {"enable_body_control": False})
        self.logger.info("Body control disabled (hands only)")
        self._last_action_label = "hands_only"
        return None

    def _draw_pose_overlay(self, frame, pose_landmarks) -> None:
        h, w = frame.shape[:2]
        pts = {}
        vis = {}
        for idx in range(min(33, len(pose_landmarks))):
            if idx < len(pose_landmarks):
                lm = pose_landmarks[idx]
                v = float(getattr(lm, "visibility", 1.0))
                vis[idx] = v
                if v > 0.4:
                    pts[idx] = (int(lm.x * w), int(lm.y * h))

        def _edge_color(s: int, e: int):
            pair = (s, e)
            rev = (e, s)
            for group_name, pairs in POSE_GROUPS.items():
                if pair in pairs or rev in pairs:
                    return POSE_GROUP_COLORS[group_name]
            return (180, 180, 180)

        for s, e in POSE_FULL_EDGES:
            if s in pts and e in pts:
                c = _edge_color(s, e)
                v = min(vis.get(s, 1.0), vis.get(e, 1.0))
                t = 1 if v < 0.65 else 2
                cv2.line(frame, pts[s], pts[e], c, t, cv2.LINE_AA)

        key_points = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}
        for idx, p in pts.items():
            v = vis.get(idx, 1.0)
            if idx in key_points:
                # subtle halo for key joints
                cv2.circle(frame, p, 6, (255, 255, 255), 1, cv2.LINE_AA)
            radius = 3 if idx in key_points else 2
            b = int(80 + 140 * max(0.0, min(1.0, v)))
            color = (b, b, 255)
            cv2.circle(frame, p, radius, color, -1, cv2.LINE_AA)

    def _draw_active_zone(self, frame) -> None:
        if self.config.ui_minimal:
            return
        if self._ui_mode != "debug":
            return
        h, w = frame.shape[:2]
        m = self._active_zone_margin
        x1, y1 = int(w * m), int(h * m)
        x2, y2 = int(w * (1.0 - m)), int(h * (1.0 - m))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1, cv2.LINE_AA)

    def _detect_hsv_fallback_point(self, frame) -> tuple[float, float] | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([110, 50, 50], dtype=np.uint8)
        upper = np.array([130, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius < 6:
            return None
        h, w = frame.shape[:2]
        nx = max(0.0, min(1.0, float(x) / max(1, w - 1)))
        ny = max(0.0, min(1.0, float(y) / max(1, h - 1)))
        return (nx, ny)

    def _apply_profile(self, profile: str) -> None:
        profile = profile.lower()
        if profile == "precision":
            self._profile_name = "precision"
            self._inference_scale = 0.72
            self._inference_stride = 1
            self._pose_inference_scale = 0.62
            self._pose_inference_stride = 2
            self._active_zone_margin = 0.05
            self.actions.mouse_smoothing_alpha = 0.28
        elif profile == "performance":
            self._profile_name = "performance"
            self._inference_scale = 0.55
            self._inference_stride = 3
            self._pose_inference_scale = 0.42
            self._pose_inference_stride = 4
            self._active_zone_margin = 0.10
            self.actions.mouse_smoothing_alpha = 0.50
        else:
            self._profile_name = "balanced"
            self._inference_scale = 0.62
            self._inference_stride = 2
            self._pose_inference_scale = 0.50
            self._pose_inference_stride = 3
            self._active_zone_margin = 0.08
            self.actions.mouse_smoothing_alpha = 0.40
        self.logger.info("Profile switched: %s", self._profile_name)

    def _estimate_gesture_confidence(self, stable_gesture: str) -> float:
        if not self._gesture_history:
            return 0.0
        same = sum(1 for g in self._gesture_history if g == stable_gesture)
        return same / len(self._gesture_history)

    def _cycle_theme(self) -> None:
        names = ["cyber", "clean", "minimal"]
        idx = names.index(self._theme_name) if self._theme_name in names else 0
        self._theme_name = names[(idx + 1) % len(names)]
        self.logger.info("Theme switched: %s", self._theme_name)

    def _toggle_ui_mode(self) -> None:
        self._ui_mode = "debug" if self._ui_mode == "pro" else "pro"
        self._last_action_label = f"ui={self._ui_mode}"
        save_config_updates(self.config_path, {"ui_mode": self._ui_mode})
        self.logger.info("UI mode switched: %s", self._ui_mode)

    def _draw_gesture_timeline(self, frame) -> None:
        if len(self._gesture_timeline) < 2:
            return
        theme = THEMES[self._theme_name]
        h, w = frame.shape[:2]
        x0 = max(10, w - 280)
        y0 = h - 26
        self._blend_roi(frame, x0, y0 - 14, x0 + 270, y0 + 6, 0.28, theme["bg"])
        recent = list(self._gesture_timeline)[-10:]
        x = x0 + 6
        for g in recent:
            c = GESTURE_ACCENT.get(g, (160, 160, 180))
            cv2.circle(frame, (x, y0), 5, c, -1, cv2.LINE_AA)
            x += 24

    def _start_calibration(self) -> None:
        self._calibrating = True
        self._calibration_started_at = time.monotonic()
        self._calib_stage_started_at = self._calibration_started_at
        self._calib_stage_index = 0
        self._calibration_samples = []
        self.logger.info("Calibration wizard started")

    def _update_calibration_wizard(self, index_point: tuple[float, float] | None) -> None:
        now = time.monotonic()
        tx, ty, _ = CALIBRATION_TARGETS[self._calib_stage_index]
        if index_point is not None:
            dx = index_point[0] - tx
            dy = index_point[1] - ty
            if (dx * dx + dy * dy) ** 0.5 <= 0.12:
                self._calibration_samples.append(index_point)
        if (now - self._calib_stage_started_at) >= self._calib_stage_seconds:
            self._calib_stage_index += 1
            self._calib_stage_started_at = now
            if self._calib_stage_index >= len(CALIBRATION_TARGETS):
                self._finish_calibration()

    def _finish_calibration(self) -> None:
        self._calibrating = False
        self._calib_stage_index = 0
        samples = self._calibration_samples
        if len(samples) < 20:
            self.logger.warning("Calibration skipped: not enough samples (%s)", len(samples))
            return

        xs = sorted(p[0] for p in samples)
        ys = sorted(p[1] for p in samples)
        n = len(samples)
        p05 = max(0, int(n * 0.05) - 1)
        p95 = min(n - 1, int(n * 0.95))
        min_x, max_x = xs[p05], xs[p95]
        min_y, max_y = ys[p05], ys[p95]
        # Build calibrated control box (with slight expansion) to better span big screens.
        expand = 0.03
        self._calib_min_x = max(0.0, min_x - expand)
        self._calib_max_x = min(1.0, max_x + expand)
        self._calib_min_y = max(0.0, min_y - expand)
        self._calib_max_y = min(1.0, max_y + expand)
        self._active_zone_margin = 0.02

        # Estimate pointer jitter and derive smoothing.
        deltas = []
        for i in range(1, n):
            dx = samples[i][0] - samples[i - 1][0]
            dy = samples[i][1] - samples[i - 1][1]
            deltas.append((dx * dx + dy * dy) ** 0.5)
        avg_delta = sum(deltas) / max(1, len(deltas))
        # Higher jitter => stronger smoothing (lower alpha).
        self.actions.mouse_smoothing_alpha = max(0.22, min(0.55, 0.50 - avg_delta * 8.0))
        self.logger.info(
            "Calibration applied: box=[%.3f..%.3f, %.3f..%.3f], smoothing_alpha=%.3f",
            self._calib_min_x,
            self._calib_max_x,
            self._calib_min_y,
            self._calib_max_y,
            self.actions.mouse_smoothing_alpha,
        )
        self._persist_calibration()

    def _map_calibrated_point(self, point: tuple[float, float] | None) -> tuple[float, float] | None:
        if point is None:
            return None
        x, y = point
        dx = max(1e-6, self._calib_max_x - self._calib_min_x)
        dy = max(1e-6, self._calib_max_y - self._calib_min_y)
        nx = (x - self._calib_min_x) / dx
        ny = (y - self._calib_min_y) / dy
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        return (nx, ny)

    def _persist_calibration(self) -> None:
        try:
            save_config_updates(
                self.config_path,
                {
                    "active_zone_margin": round(self._active_zone_margin, 4),
                    "calib_min_x": round(self._calib_min_x, 5),
                    "calib_max_x": round(self._calib_max_x, 5),
                    "calib_min_y": round(self._calib_min_y, 5),
                    "calib_max_y": round(self._calib_max_y, 5),
                    "mouse_smoothing_alpha": round(float(self.actions.mouse_smoothing_alpha), 4),
                },
            )
            self.logger.info("Calibration saved to %s", self.config_path)
        except Exception as exc:
            self.logger.warning("Calibration save failed: %s", exc)
