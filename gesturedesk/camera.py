from __future__ import annotations

import cv2


class CameraError(RuntimeError):
    """Raised when camera is unavailable."""


class CameraStream:
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 960,
        height: int = 540,
        fps: int = 30,
        fourcc: str = "MJPG",
        autofocus: bool = True,
        auto_exposure: bool = True,
        exposure: int = -1,
    ) -> None:
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise CameraError(
                f"Impossible d'ouvrir la camera id={camera_id}. Verifiez /dev/video{camera_id}."
            )
        # Best-effort capture tuning: aim for camera-native quality while keeping inference optimized elsewhere.
        if fourcc and len(fourcc) == 4:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
        if auto_exposure:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # V4L2: auto mode
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2: manual mode
            if exposure >= 0:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, int(exposure))

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        if self.cap:
            self.cap.release()
