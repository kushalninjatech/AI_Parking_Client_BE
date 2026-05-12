import logging
from typing import Optional, Tuple

import numpy as np

from app.camera.base import BaseCamera

logger = logging.getLogger(__name__)


class CSICamera(BaseCamera):
    """RPi CSI camera using picamera2 library."""

    def __init__(self, index: int = 0):
        self._index = index
        self._picam = None

    def open(self) -> bool:
        try:
            from picamera2 import Picamera2
            self._picam = Picamera2(self._index)
            config = self._picam.create_still_configuration(
                main={"size": (1920, 1080), "format": "RGB888"}
            )
            self._picam.configure(config)
            self._picam.start()
            logger.info("CSI camera %d opened", self._index)
            return True
        except Exception:
            logger.exception("Failed to open CSI camera %d", self._index)
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._picam is None:
            return False, None
        try:
            frame = self._picam.capture_array()
            # picamera2 returns RGB, convert to BGR for OpenCV
            import cv2
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame_bgr
        except Exception:
            logger.exception("Error capturing from CSI camera %d", self._index)
            return False, None

    def release(self) -> None:
        if self._picam is not None:
            self._picam.stop()
            self._picam.close()
            self._picam = None
            logger.info("CSI camera %d released", self._index)

    def is_opened(self) -> bool:
        return self._picam is not None
