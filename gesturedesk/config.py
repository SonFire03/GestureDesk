from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    camera_id: int = 0
    camera_width: int = 960
    camera_height: int = 540
    camera_fps: int = 30
    camera_fourcc: str = "MJPG"
    camera_autofocus: bool = True
    camera_auto_exposure: bool = True
    camera_exposure: int = -1
    model_path: str = "models/hand_landmarker.task"
    pose_model_path: str = "models/pose_landmarker_lite.task"
    inference_scale: float = 0.75
    inference_every_n_frames: int = 1
    pose_inference_scale: float = 0.5
    pose_inference_every_n_frames: int = 3
    draw_secondary_hand: bool = True
    draw_finger_card: bool = True
    ui_minimal: bool = False
    active_zone_margin: float = 0.08
    calib_min_x: float = 0.0
    calib_max_x: float = 1.0
    calib_min_y: float = 0.0
    calib_max_y: float = 1.0
    cooldown_seconds: float = 0.4
    click_cooldown_seconds: float = 0.4
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    mouse_smoothing_alpha: float = 0.35
    mouse_adaptive_gain_min: float = 0.7
    mouse_adaptive_gain_max: float = 2.0
    mouse_adaptive_scale_px: float = 220.0
    drag_toggle_hold_seconds: float = 0.25
    dominant_hand_mode: str = "auto"
    scroll_step: int = 120
    enable_mouse_control: bool = True
    enable_media_keys: bool = True
    enable_scroll: bool = True
    gesture_hold_seconds: float = 1.0
    enable_body_control: bool = True
    draw_pose_overlay: bool = True
    body_hold_seconds: float = 0.8


def load_config(path: str | Path = "config.json") -> AppConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return AppConfig()

    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return AppConfig(
        camera_id=int(data.get("camera_id", 0)),
        camera_width=int(data.get("camera_width", 960)),
        camera_height=int(data.get("camera_height", 540)),
        camera_fps=int(data.get("camera_fps", 30)),
        camera_fourcc=str(data.get("camera_fourcc", "MJPG")),
        camera_autofocus=bool(data.get("camera_autofocus", True)),
        camera_auto_exposure=bool(data.get("camera_auto_exposure", True)),
        camera_exposure=int(data.get("camera_exposure", -1)),
        model_path=str(data.get("model_path", "models/hand_landmarker.task")),
        pose_model_path=str(data.get("pose_model_path", "models/pose_landmarker_lite.task")),
        inference_scale=float(data.get("inference_scale", 0.75)),
        inference_every_n_frames=int(data.get("inference_every_n_frames", 1)),
        pose_inference_scale=float(data.get("pose_inference_scale", 0.5)),
        pose_inference_every_n_frames=int(data.get("pose_inference_every_n_frames", 3)),
        draw_secondary_hand=bool(data.get("draw_secondary_hand", True)),
        draw_finger_card=bool(data.get("draw_finger_card", True)),
        ui_minimal=bool(data.get("ui_minimal", False)),
        active_zone_margin=float(data.get("active_zone_margin", 0.08)),
        calib_min_x=float(data.get("calib_min_x", 0.0)),
        calib_max_x=float(data.get("calib_max_x", 1.0)),
        calib_min_y=float(data.get("calib_min_y", 0.0)),
        calib_max_y=float(data.get("calib_max_y", 1.0)),
        cooldown_seconds=float(data.get("cooldown_seconds", 0.4)),
        click_cooldown_seconds=float(data.get("click_cooldown_seconds", 0.4)),
        min_detection_confidence=float(data.get("min_detection_confidence", 0.6)),
        min_tracking_confidence=float(data.get("min_tracking_confidence", 0.6)),
        mouse_smoothing_alpha=float(data.get("mouse_smoothing_alpha", 0.35)),
        mouse_adaptive_gain_min=float(data.get("mouse_adaptive_gain_min", 0.7)),
        mouse_adaptive_gain_max=float(data.get("mouse_adaptive_gain_max", 2.0)),
        mouse_adaptive_scale_px=float(data.get("mouse_adaptive_scale_px", 220.0)),
        drag_toggle_hold_seconds=float(data.get("drag_toggle_hold_seconds", 0.25)),
        dominant_hand_mode=str(data.get("dominant_hand_mode", "auto")),
        scroll_step=int(data.get("scroll_step", 120)),
        enable_mouse_control=bool(data.get("enable_mouse_control", True)),
        enable_media_keys=bool(data.get("enable_media_keys", True)),
        enable_scroll=bool(data.get("enable_scroll", True)),
        gesture_hold_seconds=float(data.get("gesture_hold_seconds", 1.0)),
        enable_body_control=bool(data.get("enable_body_control", True)),
        draw_pose_overlay=bool(data.get("draw_pose_overlay", True)),
        body_hold_seconds=float(data.get("body_hold_seconds", 0.8)),
    )


def save_config_updates(path: str | Path, updates: dict) -> None:
    cfg_path = Path(path)
    data = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    data.update(updates)
    tmp_path = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp_path.replace(cfg_path)
