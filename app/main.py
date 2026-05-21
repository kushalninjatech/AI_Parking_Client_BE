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

# Global instances
yolo_detector = YOLODetector()
depth_estimator = DepthEstimator()
parking_detector = ParkingDetector(yolo_detector, depth_estimator)
detection_loop = DetectionLoop(parking_detector)
mqtt_publisher = MQTTPublisher(db_factory=SessionLocal)

# Latest frames per camera (for snapshot/streaming)
latest_frames = {}


def on_state_change(camera_id: int, all_results, changes):
    """Called when detection loop produces results."""
    from app.db.session import SessionLocal
    from app.models.parking_slot import ParkingSlot
    from app.models.camera import Camera

    db = SessionLocal()
    try:
        cam = db.query(Camera).filter(Camera.id == camera_id).first()
        camera_label = cam.label if cam else f"cam-{camera_id}"

        # Update slot states in local DB
        for r in all_results:
            slot = db.query(ParkingSlot).filter(ParkingSlot.id == r["id"]).first()
            if slot:
                slot.state = r["state"]
                vtype = r.get("detected_vehicle_type")
                slot.detected_vehicle_type = vtype.value if hasattr(vtype, "value") else vtype
        db.commit()
    finally:
        db.close()

    # Publish only changed slots to MQTT events topic.
    # Full-state snapshots handled by snapshot_loop on timer (MQTT_SNAPSHOT_INTERVAL).
    if changes:
        mqtt_publisher.publish_slot_events(camera_label, changes)


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
                          "detected_vehicle_type": s.detected_vehicle_type}
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
        conn.commit()

    # Load models (graceful — server starts even if models missing)
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
                 "slot_type": s.slot_type or "GENERAL"}
                for s in cam.slots if s.polygon_coords
            ]
            if slots:
                detection_loop.add_camera(cam.id, cam.source, slots, cam.camera_type)

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
