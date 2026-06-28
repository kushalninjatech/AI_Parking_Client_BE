"""AI Parking Client — Edge Device Backend.
Runs on RPi5. Manages cameras, detection, MQTT publishing.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.base import Base
from app.db.session import engine, SessionLocal
from app.detection.yolo_detector import YOLODetector
from app.detection.depth_estimator import DepthEstimator
from app.detection.detector import ParkingDetector
from app.detection.detection_loop import DetectionLoop
from app.mqtt.publisher import MQTTPublisher

# Import all models so SQLAlchemy registers them before create_all
from app.models import camera as _camera_model  # noqa: F401
from app.models import parking_slot as _slot_model  # noqa: F401
from app.models import calibration as _cal_model  # noqa: F401
from app.models import mqtt_outbox as _outbox_model  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger("ai_parking_client")

# Global instances — YOLO/Depth always created (used as fallback even in Gemini mode)
yolo_detector = YOLODetector()
depth_estimator = DepthEstimator()

# Gemini detector (only instantiated when backend=gemini)
gemini_detector = None
if settings.DETECTION_BACKEND.lower() == "gemini":
    from app.detection.gemini_detector import GeminiDetector
    gemini_detector = GeminiDetector()

parking_detector = ParkingDetector(yolo_detector, depth_estimator, gemini=gemini_detector)
detection_loop = DetectionLoop(parking_detector)
mqtt_publisher = MQTTPublisher(db_factory=SessionLocal)

# Latest frames per camera (for snapshot/streaming)
latest_frames = {}


def on_state_change(camera_id: int, all_results, changes, frame, vehicle_events=None):
    """Called when detection loop produces results."""
    import json
    from app.db.session import SessionLocal
    from app.models.parking_slot import ParkingSlot
    from app.models.camera import Camera
    from app.services.minio_service import upload_slot_image

    db = SessionLocal()
    try:
        cam = db.query(Camera).filter(Camera.id == camera_id).first()
        camera_label = cam.label if cam else f"cam-{camera_id}"

        # Build polygon lookup for changed slots (needed for cropping)
        slot_polygons = {}
        for r in changes:
            slot = db.query(ParkingSlot).filter(ParkingSlot.id == r["id"]).first()
            if slot and slot.polygon_coords:
                coords = slot.polygon_coords
                if isinstance(coords, str):
                    coords = json.loads(coords)
                slot_polygons[r["id"]] = coords

        # Update slot states in local DB
        for r in all_results:
            slot = db.query(ParkingSlot).filter(ParkingSlot.id == r["id"]).first()
            if slot:
                slot.state = r["state"]
                vtype = r.get("detected_vehicle_type")
                slot.detected_vehicle_type = vtype.value if hasattr(vtype, "value") else vtype
                slot.occupied_car = r.get("occupied_car", 0)
                slot.occupied_two_wheeler = r.get("occupied_two_wheeler", 0)
        db.commit()
    finally:
        db.close()

    # Upload cropped slot images to MinIO for changed slots
    if changes and frame is not None:
        for change in changes:
            polygon = slot_polygons.get(change["id"])
            if polygon:
                url = upload_slot_image(
                    frame, polygon, settings.DEVICE_ID, camera_label, change["label"],
                )
                if url:
                    change["image_url"] = url

    # Publish only changed slots to MQTT events topic.
    if changes:
        mqtt_publisher.publish_slot_events(camera_label, changes)

    # Publish parking scan summary (every detection cycle → Central creates history row)
    from app.core.constants import SlotState as _SS, SlotType as _ST
    db2 = SessionLocal()
    try:
        all_slots = db2.query(ParkingSlot).filter(ParkingSlot.camera_id == camera_id, ParkingSlot.is_active == True).all()
        car_occ, car_total, tw_occ, tw_total, has_obs = 0, 0, 0, 0, False
        for s in all_slots:
            cc = s.capacity_car or (1 if s.slot_type == _ST.CAR.value else 0)
            tc = s.capacity_two_wheeler or (1 if s.slot_type == _ST.TWO_WHEELER.value else 0)
            if s.slot_type == _ST.GENERAL.value and cc == 0 and tc == 0:
                cc = 1
            car_total += cc
            tw_total += tc
            car_occ += s.occupied_car or 0
            tw_occ += s.occupied_two_wheeler or 0
            if s.state == _SS.VEHICLE and (s.occupied_car or 0) == 0 and (s.occupied_two_wheeler or 0) == 0:
                if s.detected_vehicle_type == "TWO_WHEELER":
                    tw_occ += 1
                else:
                    car_occ += 1
            if s.state == _SS.OBSTRUCTED:
                has_obs = True
        img = (changes[0].get("image_url", "") if changes else "")
        mqtt_publisher.publish_parking_scan(camera_label, {
            "car_occupied": car_occ, "car_available": max(0, car_total - car_occ), "car_total": car_total,
            "two_wheeler_occupied": tw_occ, "two_wheeler_available": max(0, tw_total - tw_occ), "two_wheeler_total": tw_total,
            "has_obstruction": has_obs, "image_url": img,
        })
    finally:
        db2.close()

    # Publish vehicle entry/exit events (multi-capacity zones)
    if vehicle_events and frame is not None:
        from app.services.minio_service import upload_vehicle_crop
        for evt in vehicle_events:
            bbox = evt.get("bbox")
            if bbox:
                url = upload_vehicle_crop(
                    frame, bbox, settings.DEVICE_ID, camera_label, evt["track_id"],
                )
                if url:
                    evt["image_url"] = url
                else:
                    logger.warning("Vehicle crop upload failed for %s", evt["track_id"])
        if vehicle_events:
            mqtt_publisher.publish_vehicle_events(camera_label, vehicle_events)


def on_frame_captured(camera_id: int, frame):
    """Store latest frame for streaming/snapshot."""
    latest_frames[camera_id] = frame


async def status_loop():
    """Publish online status every 30 seconds on /status (retained)."""
    await asyncio.sleep(10)  # let MQTT finish connecting before first publish
    while True:
        mqtt_publisher.publish_status()
        await asyncio.sleep(30)


async def heartbeat_loop():
    """Publish system telemetry every 30 seconds on /heartbeat."""
    from app.utils.telemetry import collect_telemetry

    await asyncio.sleep(15)  # stagger from status_loop
    while True:
        telemetry = collect_telemetry()
        mqtt_publisher.publish_heartbeat(telemetry)
        await asyncio.sleep(30)


async def snapshot_loop():
    """Publish full slot-state snapshot per camera every MQTT_SNAPSHOT_INTERVAL seconds.

    Reconciliation: events publish on change, but if any event is lost, the snapshot
    brings central back in sync. Retained on the broker so reconnecting consumers
    immediately see current state.
    """
    await asyncio.sleep(settings.MQTT_SNAPSHOT_INTERVAL)  # delay first run
    while True:
        try:
            from app.db.session import SessionLocal
            from app.models.camera import Camera
            from app.models.parking_slot import ParkingSlot

            db = SessionLocal()
            try:
                cameras = db.query(Camera).filter(Camera.is_active == True).all()
                for cam in cameras:
                    slots = db.query(ParkingSlot).filter(ParkingSlot.camera_id == cam.id).all()
                    if not slots:
                        continue
                    mqtt_publisher.publish_slot_snapshot(
                        cam.label,
                        [{"label": s.label, "state": s.state,
                          "slot_type": s.slot_type or "GENERAL",
                          "detected_vehicle_type": s.detected_vehicle_type,
                          "occupied_car": s.occupied_car or 0,
                          "occupied_two_wheeler": s.occupied_two_wheeler or 0}
                         for s in slots],
                    )
                logger.debug("Snapshot published for %d cameras", len(cameras))
            finally:
                db.close()
        except Exception:
            logger.exception("Snapshot loop iteration failed")
        await asyncio.sleep(settings.MQTT_SNAPSHOT_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Parking Client [%s]", settings.DEVICE_ID)

    # Create tables
    Base.metadata.create_all(bind=engine)

    # SQLite doesn't add columns to existing tables via create_all — patch manually
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    cols = [c["name"] for c in insp.get_columns("parking_slots")]
    with engine.connect() as conn:
        if "slot_type" not in cols:
            conn.execute(text("ALTER TABLE parking_slots ADD COLUMN slot_type VARCHAR(20) DEFAULT 'GENERAL' NOT NULL"))
        if "detected_vehicle_type" not in cols:
            conn.execute(text("ALTER TABLE parking_slots ADD COLUMN detected_vehicle_type VARCHAR(20)"))
        if "capacity_car" not in cols:
            conn.execute(text("ALTER TABLE parking_slots ADD COLUMN capacity_car INTEGER DEFAULT 0 NOT NULL"))
        if "capacity_two_wheeler" not in cols:
            conn.execute(text("ALTER TABLE parking_slots ADD COLUMN capacity_two_wheeler INTEGER DEFAULT 0 NOT NULL"))
        if "occupied_car" not in cols:
            conn.execute(text("ALTER TABLE parking_slots ADD COLUMN occupied_car INTEGER DEFAULT 0 NOT NULL"))
        if "occupied_two_wheeler" not in cols:
            conn.execute(text("ALTER TABLE parking_slots ADD COLUMN occupied_two_wheeler INTEGER DEFAULT 0 NOT NULL"))
        conn.commit()

    # Backfill capacity for existing single-vehicle slots
    with engine.connect() as conn:
        conn.execute(text("UPDATE parking_slots SET capacity_car = 1 WHERE slot_type = 'CAR' AND capacity_car = 0"))
        conn.execute(text("UPDATE parking_slots SET capacity_two_wheeler = 1 WHERE slot_type = 'TWO_WHEELER' AND capacity_two_wheeler = 0"))
        conn.execute(text("UPDATE parking_slots SET capacity_car = 1 WHERE slot_type = 'GENERAL' AND capacity_car = 0"))
        conn.commit()

    # Load detection models based on backend
    if settings.DETECTION_BACKEND.lower() == "gemini":
        logger.info("Detection backend: GEMINI")
        if gemini_detector:
            try:
                gemini_detector.load()
            except Exception as e:
                logger.warning("Gemini detector not loaded: %s — falling back to YOLO", e)
        # Still load YOLO/Depth as fallback for debug frames and calibration
        try:
            yolo_detector.load()
        except Exception:
            pass
        try:
            depth_estimator.load()
        except Exception:
            pass
    else:
        logger.info("Detection backend: YOLO")
        try:
            yolo_detector.load()
        except Exception as e:
            logger.warning("YOLO model not loaded: %s", e)
        try:
            depth_estimator.load()
        except Exception as e:
            logger.warning("Depth model not loaded: %s", e)

    # Connect MQTT (graceful)
    try:
        mqtt_publisher.connect()
    except Exception as e:
        logger.warning("MQTT not connected: %s", e)

    # Load cameras + slots from DB and register
    from app.db.session import SessionLocal
    from app.models.camera import Camera
    db = SessionLocal()
    try:
        cameras = db.query(Camera).filter(Camera.is_active == True).all()
        for cam in cameras:
            slots = [
                {"id": s.id, "label": s.label, "polygon_coords": s.polygon_coords,
                 "slot_type": s.slot_type or "GENERAL",
                 "capacity_car": s.capacity_car or 0,
                 "capacity_two_wheeler": s.capacity_two_wheeler or 0}
                for s in cam.slots if s.polygon_coords
            ]
            if slots:
                detection_loop.add_camera(cam.id, cam.source, slots, cam.camera_type, cam.label)

                # Load calibrations
                from app.models.calibration import Calibration
                for s in cam.slots:
                    cal = db.query(Calibration).filter(Calibration.slot_id == s.id).first()
                    if cal and cal.calibration_data:
                        parking_detector.set_calibration(s.id, cal.calibration_data)
    finally:
        db.close()

    # Set callbacks
    detection_loop.set_callbacks(
        on_state_change=on_state_change,
        on_frame_captured=on_frame_captured,
    )

    # Start background tasks
    loop_task = asyncio.create_task(detection_loop.run())
    status_task = asyncio.create_task(status_loop())
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    snapshot_task = asyncio.create_task(snapshot_loop())

    logger.info("AI Parking Client ready")

    yield

    # Shutdown
    detection_loop.stop()
    loop_task.cancel()
    status_task.cancel()
    heartbeat_task.cancel()
    snapshot_task.cancel()
    mqtt_publisher.disconnect()
    logger.info("AI Parking Client shutdown")


app = FastAPI(
    title="AI Parking Client",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and register routes
from app.api.v1 import api_v1_router
app.include_router(api_v1_router)


@app.get("/health")
def health():
    return {"status": "healthy", "device_id": settings.DEVICE_ID}
