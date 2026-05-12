import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.parking_slot import ParkingSlot
from app.models.calibration import Calibration
from app.schemas.parking_slot import ParkingSlotCreate, ParkingSlotUpdate, ParkingSlotResponse

router = APIRouter(prefix="/slots", tags=["Parking Slots"])


@router.post("", response_model=ParkingSlotResponse, status_code=201)
def create_slot(body: ParkingSlotCreate, db: Session = Depends(get_db)):
    # Auto-calculate bounding box from polygon
    pos_x1, pos_y1, pos_x2, pos_y2 = None, None, None, None
    if body.polygon_coords:
        try:
            points = json.loads(body.polygon_coords)
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            pos_x1, pos_y1 = min(xs), min(ys)
            pos_x2, pos_y2 = max(xs), max(ys)
        except (json.JSONDecodeError, IndexError):
            pass

    slot = ParkingSlot(
        label=body.label,
        camera_id=body.camera_id,
        polygon_coords=body.polygon_coords,
        pos_x1=pos_x1, pos_y1=pos_y1,
        pos_x2=pos_x2, pos_y2=pos_y2,
    )
    db.add(slot)
    db.commit()
    db.refresh(slot)

    # Update detection loop
    _update_detection_loop(slot.camera_id, db)

    return slot


@router.get("", response_model=List[ParkingSlotResponse])
def list_slots(camera_id: int = None, db: Session = Depends(get_db)):
    query = db.query(ParkingSlot)
    if camera_id:
        query = query.filter(ParkingSlot.camera_id == camera_id)
    return query.all()


@router.get("/{slot_id}", response_model=ParkingSlotResponse)
def get_slot(slot_id: int, db: Session = Depends(get_db)):
    slot = db.query(ParkingSlot).filter(ParkingSlot.id == slot_id).first()
    if not slot:
        raise HTTPException(404, "Slot not found")
    return slot


@router.patch("/{slot_id}", response_model=ParkingSlotResponse)
def update_slot(slot_id: int, body: ParkingSlotUpdate, db: Session = Depends(get_db)):
    slot = db.query(ParkingSlot).filter(ParkingSlot.id == slot_id).first()
    if not slot:
        raise HTTPException(404, "Slot not found")

    data = body.model_dump(exclude_unset=True)

    # Recalculate bbox if polygon changed
    if "polygon_coords" in data and data["polygon_coords"]:
        try:
            points = json.loads(data["polygon_coords"])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            data["pos_x1"], data["pos_y1"] = min(xs), min(ys)
            data["pos_x2"], data["pos_y2"] = max(xs), max(ys)
        except (json.JSONDecodeError, IndexError):
            pass

    for k, v in data.items():
        setattr(slot, k, v)
    db.commit()
    db.refresh(slot)

    _update_detection_loop(slot.camera_id, db)

    return slot


@router.delete("/{slot_id}")
def delete_slot(slot_id: int, db: Session = Depends(get_db)):
    slot = db.query(ParkingSlot).filter(ParkingSlot.id == slot_id).first()
    if not slot:
        raise HTTPException(404, "Slot not found")
    camera_id = slot.camera_id
    db.delete(slot)
    db.commit()
    _update_detection_loop(camera_id, db)
    return {"message": "Slot deleted"}


@router.post("/{slot_id}/calibrate")
def calibrate_slot(slot_id: int, db: Session = Depends(get_db)):
    """Calibrate a slot with the latest camera frame (empty reference)."""
    slot = db.query(ParkingSlot).filter(ParkingSlot.id == slot_id).first()
    if not slot or not slot.polygon_coords:
        raise HTTPException(400, "Slot not found or no polygon defined")

    # Get latest frame from detection loop
    from app.main import latest_frames, parking_detector
    frame = latest_frames.get(slot.camera_id)
    if frame is None:
        raise HTTPException(400, "No frame available — ensure camera is capturing")

    polygon = json.loads(slot.polygon_coords)
    cal_data = parking_detector.calibrate_slot(frame, slot.id, polygon)

    if not cal_data:
        raise HTTPException(500, "Calibration failed")

    # Save to DB
    existing = db.query(Calibration).filter(Calibration.slot_id == slot_id).first()
    if existing:
        existing.calibration_data = cal_data
    else:
        db.add(Calibration(slot_id=slot_id, calibration_data=cal_data))
    db.commit()

    # Update detector
    parking_detector.set_calibration(slot.id, cal_data)

    return {"message": f"Slot {slot.label} calibrated"}


def _update_detection_loop(camera_id: int, db: Session) -> None:
    """Refresh detection loop with updated slots for a camera."""
    from app.main import detection_loop
    slots = db.query(ParkingSlot).filter(ParkingSlot.camera_id == camera_id).all()
    slot_dicts = [
        {"id": s.id, "label": s.label, "polygon_coords": s.polygon_coords}
        for s in slots
    ]
    detection_loop.update_slots(camera_id, slot_dicts)
