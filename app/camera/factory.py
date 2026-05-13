import logging
from typing import Optional

from app.camera.base import BaseCamera
from app.camera.opencv_camera import OpenCVCamera

logger = logging.getLogger(__name__)


def create_camera(source: str, camera_type: Optional[str] = None) -> BaseCamera:
    """Create camera based on source string and optional camera_type hint.

    Supported formats:
      - "csi://0"        → CSI camera (RPi camera module)
      - "rtsp://..."     → IP camera via RTSP
      - "/dev/video0"    → USB webcam
      - "0"              → USB webcam by index (or CSI index if camera_type="CSI")
    """
    is_csi = source.startswith("csi://") or (camera_type and camera_type.upper() == "CSI")

    if is_csi:
        # Resolve the numeric index from either "csi://0" or bare "0"
        raw_index = source.replace("csi://", "") if source.startswith("csi://") else source
        try:
            index = int(raw_index)
        except ValueError:
            index = 0
        try:
            from app.camera.csi_camera import CSICamera
            return CSICamera(index)
        except ImportError:
            logger.warning("picamera2 not available, using OpenCV for CSI index %d", index)
            return OpenCVCamera(str(index))

    # RTSP or USB — both use OpenCV
    return OpenCVCamera(source)
