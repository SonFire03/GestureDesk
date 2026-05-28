from __future__ import annotations

from pathlib import Path

from gesturedesk.config import AppConfig
from gesturedesk.diagnostics import check_display_env, check_model_path


def test_check_model_path_ok(tmp_path: Path):
    model = tmp_path / "hand_landmarker.task"
    model.write_text("x", encoding="utf-8")
    config = AppConfig(model_path=str(model))
    result = check_model_path(config)
    assert result.ok is True


def test_check_model_path_missing(tmp_path: Path):
    model = tmp_path / "missing.task"
    config = AppConfig(model_path=str(model))
    result = check_model_path(config)
    assert result.ok is False


def test_check_display_env(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    result = check_display_env()
    assert result.ok is True
