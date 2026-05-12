from sqlalchemy import Column, String, Integer, ForeignKey, LargeBinary

from app.db.base import Base, TimestampMixin


class Calibration(Base, TimestampMixin):
    """Per-slot depth calibration data (empty reference)."""
    __tablename__ = "calibrations"

    slot_id = Column(Integer, ForeignKey("parking_slots.id"), nullable=False, unique=True)

    # Serialized numpy arrays (reference depth stats per grid cell)
    calibration_data = Column(LargeBinary, nullable=True)

    # Reference snapshot path
    reference_image_path = Column(String(500), nullable=True)
