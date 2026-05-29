from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

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

def check_pose_model_path(config: AppConfig) -> CheckResult:
    if not config.enable_body_control:
        return CheckResult(True, "Body control desactive: modele pose non requis")
    model = Path(config.pose_model_path)
    if model.exists() and model.is_file():
        return CheckResult(True, f"Modele pose OK: {model}")
    return CheckResult(False, f"Modele pose manquant: {model}")


def check_camera_access(camera_id: int) -> CheckResult:
    dev_path = Path(f"/dev/video{camera_id}")
    if not dev_path.exists():
        return CheckResult(False, f"Camera indisponible: id={camera_id} ({dev_path} absent)")

    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            return CheckResult(False, f"Camera indisponible: id={camera_id} (/dev/video{camera_id})")
        start = monotonic()
        ok = False
        while monotonic() - start < 1.2:
            ok, _ = cap.read()
            if ok:
                break
        if not ok:
            return CheckResult(False, f"Camera ouverte mais lecture frame echouee (timeout): id={camera_id}")
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
        check_pose_model_path(config),
        check_camera_access(config.camera_id),
        check_display_env(),
    ]


def list_camera_candidates(max_id: int = 12) -> list[tuple[int, bool, bool]]:
    out: list[tuple[int, bool, bool]] = []
    for i in range(max_id):
        dev_path = Path(f"/dev/video{i}")
        if not dev_path.exists():
            continue
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        try:
            opened = cap.isOpened()
            read_ok = False
            if opened:
                read_ok, _ = cap.read()
            out.append((i, opened, read_ok))
        finally:
            cap.release()
    return out
