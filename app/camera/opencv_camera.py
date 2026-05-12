import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from app.camera.base import BaseCamera

logger = logging.getLogger(__name__)


class OpenCVCamera(BaseCamera):
    """Camera using OpenCV — supports USB webcams and RTSP streams."""

    def __init__(self, source: str):
        self._source = source
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        try:
            # Integer source (USB device index)
            if self._source.isdigit():
                self._cap = cv2.VideoCapture(int(self._source))
            else:
                self._cap = cv2.VideoCapture(self._source)

            if not self._cap.isOpened():
                logger.error("Failed to open camera: %s", self._source)
                return False

            logger.info("Camera opened: %s", self._source)
            return True
        except Exception:
            logger.exception("Error opening camera: %s", self._source)
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released: %s", self._source)

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
