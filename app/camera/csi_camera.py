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
            # create_video_configuration is correct for continuous capture_array() calls.
            # BGR888 gives a native BGR array — no colour conversion needed.
            config = self._picam.create_video_configuration(
                main={"size": (1920, 1080), "format": "BGR888"}
            )
            self._picam.configure(config)
            self._picam.start()
            logger.info("CSI camera %d opened (BGR888 video config)", self._index)
            return True
        except Exception:
            logger.exception("Failed to open CSI camera %d", self._index)
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._picam is None:
            return False, None
        try:
            frame = self._picam.capture_array()
            # capture_array() returns BGR888 — ready for OpenCV, no conversion needed
            return True, frame
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
