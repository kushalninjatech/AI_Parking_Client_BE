import abc
from typing import Optional, Tuple

import numpy as np


class BaseCamera(abc.ABC):
    """Abstract camera interface."""

    @abc.abstractmethod
    def open(self) -> bool:
        """Open camera connection."""
        ...

    @abc.abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture a single frame. Returns (success, frame_bgr)."""
        ...

    @abc.abstractmethod
    def release(self) -> None:
        """Release camera resources."""
        ...

    @abc.abstractmethod
    def is_opened(self) -> bool:
        """Check if camera is open."""
        ...

    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Get (width, height) of captured frame."""
        ok, frame = self.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            return w, h
        return 0, 0
