from sqlalchemy import Column, String, Float, Boolean, Enum as SAEnum
from sqlalchemy.orm import relationship

from app.core.constants import CameraType, CameraStatus
from app.db.base import Base, TimestampMixin


class Camera(Base, TimestampMixin):
    __tablename__ = "cameras"

    label = Column(String(100), nullable=False)
    source = Column(String(500), nullable=False)  # "csi://0", "rtsp://...", "/dev/video0"
    camera_type = Column(SAEnum(CameraType), nullable=False, default=CameraType.USB)
    status = Column(SAEnum(CameraStatus), nullable=False, default=CameraStatus.ACTIVE)
    detection_interval = Column(Float, nullable=False, default=30.0)
    is_active = Column(Boolean, nullable=False, default=True)
    worker_running = Column(Boolean, nullable=False, default=False)

    # Frame dimensions (set on first capture)
    frame_width = Column(Float, nullable=True)
    frame_height = Column(Float, nullable=True)

    # Reference snapshot path (for polygon drawing)
    reference_snapshot_path = Column(String(500), nullable=True)

    # Central IDs (for syncing with central platform)
    central_camera_id = Column(String(100), nullable=True)
    central_device_id = Column(String(100), nullable=True)

    slots = relationship("ParkingSlot", back_populates="camera", lazy="selectin")
