from __future__ import annotations

import cv2


class CameraError(RuntimeError):
    """Raised when camera is unavailable."""


class CameraStream:
    def __init__(self, camera_id: int = 0, width: int = 960, height: int = 540, fps: int = 30) -> None:
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise CameraError(
                f"Impossible d'ouvrir la camera id={camera_id}. Verifiez /dev/video{camera_id}."
            )
        # Best-effort capture size tuning to reduce processing load.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        if self.cap:
            self.cap.release()
