import json
import os

import cv2
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.camera import Camera
from app.schemas.camera import CameraCreate, CameraUpdate, CameraResponse
from app.camera.factory import create_camera

router = APIRouter(prefix="/cameras", tags=["Cameras"])


@router.post("", response_model=CameraResponse, status_code=201)
def create_camera_endpoint(body: CameraCreate, db: Session = Depends(get_db)):
    cam = Camera(
        label=body.label,
        source=body.source,
        camera_type=body.camera_type,
        detection_interval=body.detection_interval,
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)

    # Register on central platform
    from app.services.central_sync import central_sync
    from app.core.config import settings
    central_cam_id = central_sync.register_camera({
        "label": cam.label,
        "source": cam.source,
        "camera_type": cam.camera_type,
    })
    if central_cam_id:
        cam.central_camera_id = central_cam_id
        cam.central_device_id = settings.DEVICE_ID
        db.commit()
        db.refresh(cam)

    return cam


@router.get("", response_model=list[CameraResponse])
def list_cameras(db: Session = Depends(get_db)):
    return db.query(Camera).filter(Camera.is_active == True).all()


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(camera_id: int, db: Session = Depends(get_db)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")
    return cam


@router.patch("/{camera_id}", response_model=CameraResponse)
def update_camera(camera_id: int, body: CameraUpdate, db: Session = Depends(get_db)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(cam, k, v)
    db.commit()
    db.refresh(cam)
    return cam


@router.delete("/{camera_id}")
def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")
    cam.is_active = False
    db.commit()
    from app.main import detection_loop
    detection_loop.remove_camera(camera_id)
    return {"message": "Camera deleted"}


@router.post("/{camera_id}/snapshot")
def capture_snapshot(camera_id: int, db: Session = Depends(get_db)):
    """Capture a reference snapshot for polygon drawing."""
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")

    camera = create_camera(cam.source, cam.camera_type)
    if not camera.open():
        raise HTTPException(500, "Failed to open camera")

    ok, frame = camera.read()
    camera.release()

    if not ok or frame is None:
        raise HTTPException(500, "Failed to capture frame")

    # Save snapshot
    os.makedirs("data/snapshots", exist_ok=True)
    path = f"data/snapshots/camera_{camera_id}_ref.jpg"
    cv2.imwrite(path, frame)

    # Update camera record
    h, w = frame.shape[:2]
    cam.reference_snapshot_path = path
    cam.frame_width = w
    cam.frame_height = h
    db.commit()

    return {"path": path, "width": w, "height": h}


@router.get("/{camera_id}/snapshot")
def get_snapshot(camera_id: int, db: Session = Depends(get_db)):
    """Get the reference snapshot as JPEG."""
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam or not cam.reference_snapshot_path:
        raise HTTPException(404, "No snapshot available")

    with open(cam.reference_snapshot_path, "rb") as f:
        return Response(content=f.read(), media_type="image/jpeg")


@router.get("/{camera_id}/latest-frame")
def get_latest_frame(camera_id: int):
    """Get the latest captured frame (from detection loop)."""
    from app.main import latest_frames
    frame = latest_frames.get(camera_id)
    if frame is None:
        raise HTTPException(404, "No frame available")

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@router.get("/{camera_id}/live-frame")
def get_live_frame(camera_id: int, db: Session = Depends(get_db)):
    """Capture a single live frame directly from the camera (no 30s wait)."""
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")

    camera = create_camera(cam.source, cam.camera_type)
    if not camera.open():
        raise HTTPException(500, "Failed to open camera")

    ok, frame = camera.read()
    camera.release()

    if not ok or frame is None:
        raise HTTPException(500, "Failed to capture frame")

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")
