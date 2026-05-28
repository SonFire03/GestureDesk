from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2

from gesturedesk.config import AppConfig


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    message: str


def check_model_path(config: AppConfig) -> CheckResult:
    model = Path(config.model_path)
    if model.exists() and model.is_file():
        return CheckResult(True, f"Modele OK: {model}")
    return CheckResult(False, f"Modele manquant: {model}")


def check_camera_access(camera_id: int) -> CheckResult:
    cap = cv2.VideoCapture(camera_id)
    try:
        if not cap.isOpened():
            return CheckResult(False, f"Camera indisponible: id={camera_id} (/dev/video{camera_id})")
        ok, _ = cap.read()
        if not ok:
            return CheckResult(False, f"Camera ouverte mais lecture frame echouee: id={camera_id}")
        return CheckResult(True, f"Camera OK: id={camera_id}")
    finally:
        cap.release()


def check_display_env() -> CheckResult:
    wayland = os.environ.get("WAYLAND_DISPLAY")
    x11 = os.environ.get("DISPLAY")
    if wayland:
        return CheckResult(True, "Session Wayland detectee (PyAutoGUI peut etre limite)")
    if x11:
        return CheckResult(True, "Session X11 detectee")
    return CheckResult(False, "Aucune session graphique detectee (DISPLAY/WAYLAND_DISPLAY absents)")


def run_preflight_checks(config: AppConfig) -> list[CheckResult]:
    return [
        check_model_path(config),
        check_camera_access(config.camera_id),
        check_display_env(),
    ]
