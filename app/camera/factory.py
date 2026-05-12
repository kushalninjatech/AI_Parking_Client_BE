import logging

from app.camera.base import BaseCamera
from app.camera.opencv_camera import OpenCVCamera

logger = logging.getLogger(__name__)


def create_camera(source: str) -> BaseCamera:
    """Create camera based on source string.

    Supported formats:
      - "csi://0"        → CSI camera (RPi camera module)
      - "rtsp://..."     → IP camera via RTSP
      - "/dev/video0"    → USB webcam
      - "0"              → USB webcam by index
    """
    if source.startswith("csi://"):
        # CSI camera — try picamera2 first, fall back to OpenCV
        try:
            from app.camera.csi_camera import CSICamera
            return CSICamera(int(source.replace("csi://", "")))
        except ImportError:
            logger.warning("picamera2 not available, using OpenCV for CSI")
            index = int(source.replace("csi://", ""))
            return OpenCVCamera(str(index))

    # RTSP or USB — both use OpenCV
    return OpenCVCamera(source)
