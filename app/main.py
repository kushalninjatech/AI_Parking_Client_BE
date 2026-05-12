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
from app.db.session import engine
from app.detection.yolo_detector import YOLODetector
from app.detection.depth_estimator import DepthEstimator
from app.detection.detector import ParkingDetector
from app.detection.detection_loop import DetectionLoop
from app.mqtt.publisher import MQTTPublisher

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger("ai_parking_client")

# Global instances
yolo_detector = YOLODetector()
depth_estimator = DepthEstimator()
parking_detector = ParkingDetector(yolo_detector, depth_estimator)
detection_loop = DetectionLoop(parking_detector)
mqtt_publisher = MQTTPublisher()

# Latest frames per camera (for snapshot/streaming)
latest_frames = {}


def on_state_change(camera_id: int, all_results, changes):
    """Called when detection loop produces results."""
    from app.db.session import SessionLocal
    from app.models.parking_slot import ParkingSlot

    # Get camera label
    db = SessionLocal()
    try:
        from app.models.camera import Camera
        cam = db.query(Camera).filter(Camera.id == camera_id).first()
        camera_label = cam.label if cam else f"cam-{camera_id}"

        # Update slot states in local DB
        for r in all_results:
            slot = db.query(ParkingSlot).filter(ParkingSlot.id == r["id"]).first()
            if slot:
                slot.state = r["state"]
        db.commit()
    finally:
        db.close()

    # Publish to MQTT (all results, not just changes)
    mqtt_publisher.publish_slot_states(camera_label, all_results)


def on_frame_captured(camera_id: int, frame):
    """Store latest frame for streaming/snapshot."""
    latest_frames[camera_id] = frame


async def heartbeat_loop():
    """Publish heartbeat every 30 seconds."""
    while True:
        mqtt_publisher.publish_heartbeat()
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Parking Client [%s]", settings.DEVICE_ID)

    # Create tables
    Base.metadata.create_all(bind=engine)

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
                {"id": s.id, "label": s.label, "polygon_coords": s.polygon_coords}
                for s in cam.slots if s.polygon_coords
            ]
            if slots:
                detection_loop.add_camera(cam.id, cam.source, slots)

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

    # Start detection loop + heartbeat as background tasks
    loop_task = asyncio.create_task(detection_loop.run())
    heartbeat_task = asyncio.create_task(heartbeat_loop())

    logger.info("AI Parking Client ready")

    yield

    # Shutdown
    detection_loop.stop()
    loop_task.cancel()
    heartbeat_task.cancel()
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
