from typing import Optional

from pydantic import BaseModel

from app.core.constants import CameraType, CameraStatus


class CameraCreate(BaseModel):
    label: str
    source: str
    camera_type: CameraType = CameraType.USB
    detection_interval: float = 30.0


class CameraUpdate(BaseModel):
    label: Optional[str] = None
    source: Optional[str] = None
    detection_interval: Optional[float] = None
    is_active: Optional[bool] = None


class CameraResponse(BaseModel):
    id: int
    label: str
    source: str
    camera_type: str
    status: str
    detection_interval: float
    is_active: bool
    worker_running: bool
    frame_width: Optional[float]
    frame_height: Optional[float]
    reference_snapshot_path: Optional[str]

    class Config:
        from_attributes = True
