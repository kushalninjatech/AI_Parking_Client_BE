from typing import Optional

from pydantic import BaseModel

from app.core.constants import SlotState


class ParkingSlotCreate(BaseModel):
    label: str
    camera_id: int
    polygon_coords: Optional[str] = None  # JSON: [[x1,y1],[x2,y2],...]


class ParkingSlotUpdate(BaseModel):
    label: Optional[str] = None
    polygon_coords: Optional[str] = None
    state: Optional[SlotState] = None


class ParkingSlotResponse(BaseModel):
    id: int
    label: str
    camera_id: int
    state: str
    polygon_coords: Optional[str]
    pos_x1: Optional[int]
    pos_y1: Optional[int]
    pos_x2: Optional[int]
    pos_y2: Optional[int]
    central_slot_id: Optional[str] = None

    class Config:
        from_attributes = True
