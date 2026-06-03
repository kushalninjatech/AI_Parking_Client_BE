from typing import Optional

from pydantic import BaseModel

from app.core.constants import SlotState, SlotType


class ParkingSlotCreate(BaseModel):
    label: str
    camera_id: int
    polygon_coords: Optional[str] = None  # JSON: [[x1,y1],[x2,y2],...]
    slot_type: SlotType = SlotType.GENERAL
    capacity_car: int = 0
    capacity_two_wheeler: int = 0


class ParkingSlotUpdate(BaseModel):
    label: Optional[str] = None
    polygon_coords: Optional[str] = None
    state: Optional[SlotState] = None
    slot_type: Optional[SlotType] = None
    capacity_car: Optional[int] = None
    capacity_two_wheeler: Optional[int] = None


class ParkingSlotResponse(BaseModel):
    id: int
    label: str
    camera_id: int
    state: str
    slot_type: str
    detected_vehicle_type: Optional[str] = None
    capacity_car: int = 0
    capacity_two_wheeler: int = 0
    occupied_car: int = 0
    occupied_two_wheeler: int = 0
    polygon_coords: Optional[str]
    pos_x1: Optional[int]
    pos_y1: Optional[int]
    pos_x2: Optional[int]
    pos_y2: Optional[int]
    central_slot_id: Optional[str] = None

    class Config:
        from_attributes = True
