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


# YOLO COCO class IDs — swapped to person (0) for testing; restore to {2, 3, 5, 7} for production
VEHICLE_CLASS_IDS = {0}  # person
