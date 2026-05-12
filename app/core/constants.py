import enum


class SlotState(str, enum.Enum):
    VEHICLE = "VEHICLE"
    EMPTY = "EMPTY"
    OBSTRUCTED = "OBSTRUCTED"


class CameraType(str, enum.Enum):
    CSI = "CSI"
    RTSP = "RTSP"
    USB = "USB"


class CameraStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    FAILED = "FAILED"


# YOLO COCO vehicle class IDs
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck
