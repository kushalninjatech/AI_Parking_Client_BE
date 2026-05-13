import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from app.camera.base import BaseCamera

logger = logging.getLogger(__name__)

_RTSP_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")


class OpenCVCamera(BaseCamera):
    """Camera using OpenCV — supports USB webcams and RTSP/IP streams."""

    def __init__(self, source: str):
        self._source = source
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_rtsp = source.startswith(_RTSP_PREFIXES)

    def open(self) -> bool:
        try:
            if self._source.isdigit():
                self._cap = cv2.VideoCapture(int(self._source))
            else:
                self._cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)

            if not self._cap.isOpened():
                logger.error("Failed to open camera: %s", self._source)
                return False

            if self._is_rtsp:
                # Keep only the latest frame — avoids reading stale buffered frames
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            logger.info("Camera opened: %s", self._source)
            return True
        except Exception:
            logger.exception("Error opening camera: %s", self._source)
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None

        if self._is_rtsp:
            # Discard buffered frames so we always get the current one
            for _ in range(4):
                self._cap.grab()

        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released: %s", self._source)

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
