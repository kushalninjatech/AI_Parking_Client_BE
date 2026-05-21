from sqlalchemy import Column, String, Integer, Text, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import relationship

from app.core.constants import SlotState, SlotType
from app.db.base import Base, TimestampMixin


class ParkingSlot(Base, TimestampMixin):
    __tablename__ = "parking_slots"

    label = Column(String(50), nullable=False)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    state = Column(SAEnum(SlotState), nullable=False, default=SlotState.EMPTY)
    slot_type = Column(String(20), nullable=False, default=SlotType.GENERAL.value)
    detected_vehicle_type = Column(String(20), nullable=True)

    # Polygon ROI coordinates (JSON: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    polygon_coords = Column(Text, nullable=True)

    # Bounding box (auto-calculated from polygon)
    pos_x1 = Column(Integer, nullable=True)
    pos_y1 = Column(Integer, nullable=True)
    pos_x2 = Column(Integer, nullable=True)
    pos_y2 = Column(Integer, nullable=True)

    # Central slot ID (for syncing)
    central_slot_id = Column(String(100), nullable=True)

    camera = relationship("Camera", back_populates="slots", lazy="selectin")
