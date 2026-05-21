import enum
from typing import Dict


class SlotState(str, enum.Enum):
    VEHICLE = "VEHICLE"
    EMPTY = "EMPTY"
    OBSTRUCTED = "OBSTRUCTED"


class SlotType(str, enum.Enum):
    CAR = "CAR"
    TWO_WHEELER = "TWO_WHEELER"
    GENERAL = "GENERAL"


class VehicleType(str, enum.Enum):
    CAR = "CAR"
    TWO_WHEELER = "TWO_WHEELER"


class CameraType(str, enum.Enum):
    CSI = "CSI"
    RTSP = "RTSP"
    USB = "USB"


class CameraStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    FAILED = "FAILED"


# COCO class_id → VehicleType mapping
# 0=person (TESTING), 2=car, 3=motorcycle
COCO_VEHICLE_MAP: Dict[int, VehicleType] = {
    0: VehicleType.CAR,           # person — TESTING ONLY
    2: VehicleType.CAR,           # car
    3: VehicleType.TWO_WHEELER,   # motorcycle
}

# Priority for "largest vehicle wins" when multiple overlap a slot polygon
VEHICLE_PRIORITY: Dict[VehicleType, int] = {
    VehicleType.TWO_WHEELER: 1,
    VehicleType.CAR: 2,
}
