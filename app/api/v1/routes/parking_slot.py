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

    _update_detection_loop(slot.camera_id, db)
    _sync_slots_to_central(slot.camera_id, db)

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
    _sync_slots_to_central(slot.camera_id, db)

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
    _sync_slots_to_central(camera_id, db)
    return {"message": "Slot deleted"}


@router.post("/{slot_id}/calibrate")
def calibrate_slot(slot_id: int, db: Session = Depends(get_db)):
    """Calibrate a slot using a live camera frame as the empty reference."""
    slot = db.query(ParkingSlot).filter(ParkingSlot.id == slot_id).first()
    if not slot or not slot.polygon_coords:
        raise HTTPException(400, "Slot not found or no polygon defined")

    from app.main import parking_detector
    from app.models.camera import Camera
    from app.camera.factory import create_camera

    # Always capture a fresh frame so calibration reflects the current (empty) slot state
    cam_record = db.query(Camera).filter(Camera.id == slot.camera_id).first()
    if not cam_record:
        raise HTTPException(400, "Camera not found")
    camera = create_camera(cam_record.source, cam_record.camera_type)
    if not camera.open():
        raise HTTPException(500, "Failed to open camera for calibration")
    ok, frame = camera.read()
    camera.release()
    if not ok or frame is None:
        raise HTTPException(500, "Failed to capture frame for calibration")

    polygon = json.loads(slot.polygon_coords)
    cal_data = parking_detector.calibrate_slot(frame, slot.id, polygon)

    if not cal_data:
        raise HTTPException(500, "Calibration failed — ensure depth model is loaded")

    existing = db.query(Calibration).filter(Calibration.slot_id == slot_id).first()
    if existing:
        existing.calibration_data = cal_data
    else:
        db.add(Calibration(slot_id=slot_id, calibration_data=cal_data))
    db.commit()

    parking_detector.set_calibration(slot.id, cal_data)
    return {"message": f"Slot {slot.label} calibrated successfully"}


def _sync_slots_to_central(camera_id: int, db: Session) -> None:
    """Publish all slots for a camera to Central via MQTT (upsert)."""
    from app.main import mqtt_publisher
    from app.models.camera import Camera

    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        return

    slots = db.query(ParkingSlot).filter(ParkingSlot.camera_id == camera_id).all()
    mqtt_publisher.publish_sync_slots("upsert", cam.label, [
        {
            "label": s.label,
            "polygon_coords": s.polygon_coords,
            "pos_x1": s.pos_x1, "pos_y1": s.pos_y1,
            "pos_x2": s.pos_x2, "pos_y2": s.pos_y2,
        }
        for s in slots
    ])


def _update_detection_loop(camera_id: int, db: Session) -> None:
    """Register or refresh camera + slots in the detection loop."""
    from app.main import detection_loop
    from app.models.camera import Camera

    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        return

    slots = db.query(ParkingSlot).filter(ParkingSlot.camera_id == camera_id).all()
    slot_dicts = [
        {"id": s.id, "label": s.label, "polygon_coords": s.polygon_coords}
        for s in slots
    ]
    # add_camera registers new cameras AND updates existing ones
    detection_loop.add_camera(camera_id, cam.source, slot_dicts, cam.camera_type)
