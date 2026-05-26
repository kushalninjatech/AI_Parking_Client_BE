"""Detection loop — runs sequentially for all cameras at configured interval."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.camera.factory import create_camera
from app.core.config import settings
from app.core.constants import SlotState, VehicleType
from app.detection.detector import ParkingDetector

logger = logging.getLogger(__name__)

FRAME_CAPTURE_INTERVAL = 3  # seconds — lightweight frame grab for dashboard display


class DetectionLoop:
    """Main detection loop. Processes all cameras sequentially."""

    def __init__(self, detector: ParkingDetector) -> None:
        self._detector = detector
        self._camera_configs: Dict[int, Dict] = {}
        self._running = False
        self._slot_states: Dict[int, SlotState] = {}
        self._slot_vehicle_types: Dict[int, Optional[VehicleType]] = {}
        self._on_state_change = None
        self._on_frame_captured = None
        self._last_detection_time: Dict[int, float] = {}  # camera_id → last full-detection time

    def set_callbacks(self, on_state_change=None, on_frame_captured=None) -> None:
        self._on_state_change = on_state_change
        self._on_frame_captured = on_frame_captured

    def add_camera(self, camera_id: int, source: str, slots: List[Dict], camera_type: str = None) -> None:
        self._camera_configs[camera_id] = {"source": source, "slots": slots, "camera_type": camera_type}
        logger.info("Camera %d registered with %d slots", camera_id, len(slots))

    def remove_camera(self, camera_id: int) -> None:
        self._camera_configs.pop(camera_id, None)
        logger.info("Camera %d removed", camera_id)

    def update_slots(self, camera_id: int, slots: List[Dict]) -> None:
        if camera_id in self._camera_configs:
            self._camera_configs[camera_id]["slots"] = slots

    async def run(self) -> None:
        self._running = True
        logger.info(
            "Detection loop started (detection=%ds, frame_capture=%ds)",
            settings.DETECTION_INTERVAL, FRAME_CAPTURE_INTERVAL,
        )

        while self._running:
            cycle_start = time.time()
            now = cycle_start

            for camera_id, config in list(self._camera_configs.items()):
                try:
                    run_detection = (
                        now - self._last_detection_time.get(camera_id, 0)
                    ) >= settings.DETECTION_INTERVAL
                    await self._process_camera(camera_id, config, run_detection=run_detection)
                    if run_detection:
                        self._last_detection_time[camera_id] = now
                except Exception:
                    logger.exception("Error processing camera %d", camera_id)

            elapsed = time.time() - cycle_start
            sleep_time = max(0, FRAME_CAPTURE_INTERVAL - elapsed)
            await asyncio.sleep(sleep_time)

    def stop(self) -> None:
        self._running = False
        logger.info("Detection loop stopped")

    # ------------------------------------------------------------------
    # Blocking helpers — called via asyncio.to_thread so they never
    # block the FastAPI event loop
    # ------------------------------------------------------------------

    @staticmethod
    def _capture_frame(
        source: str, camera_type: Optional[str], camera_id: int
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Open camera, grab one frame, release. Runs in a thread."""
        cam = create_camera(source, camera_type)
        if not cam.open():
            logger.error("Failed to open camera %d (%s)", camera_id, source)
            return False, None
        ok, frame = cam.read()
        cam.release()
        if not ok or frame is None:
            logger.warning("Failed to capture from camera %d", camera_id)
        return ok, frame

    def _run_detection(self, frame: np.ndarray, slot_dicts: List[Dict]) -> List[Dict]:
        """Run YOLO + depth inference. Runs in a thread."""
        return self._detector.detect_frame(frame, slot_dicts)

    # ------------------------------------------------------------------

    async def _process_camera(self, camera_id: int, config: Dict, run_detection: bool = True) -> None:
        source = config["source"]
        slots = config["slots"]
        camera_type = config.get("camera_type")

        # Always capture a frame (updates dashboard display every 3s)
        ok, frame = await asyncio.to_thread(
            self._capture_frame, source, camera_type, camera_id
        )
        if not ok or frame is None:
            return

        if self._on_frame_captured:
            self._on_frame_captured(camera_id, frame)

        # Only run heavy inference on the detection interval
        if not run_detection:
            return

        slot_dicts = [
            {"id": s["id"], "label": s["label"], "polygon_coords": s["polygon_coords"],
             "slot_type": s.get("slot_type", "GENERAL")}
            for s in slots if s.get("polygon_coords")
        ]
        if not slot_dicts:
            return

        results = await asyncio.to_thread(self._run_detection, frame, slot_dicts)

        changes = []
        for r in results:
            slot_id = r["id"]
            new_state = r["state"]
            new_vtype = r.get("detected_vehicle_type")
            old_state = self._slot_states.get(slot_id)
            old_vtype = self._slot_vehicle_types.get(slot_id)
            if old_state != new_state or old_vtype != new_vtype:
                self._slot_states[slot_id] = new_state
                self._slot_vehicle_types[slot_id] = new_vtype
                changes.append(r)
                logger.info(
                    "Slot %s: %s(%s) → %s(%s) (conf=%.2f)",
                    r["label"], old_state, old_vtype, new_state, new_vtype, r["confidence"],
                )

        if self._on_state_change and results:
            self._on_state_change(camera_id, results, changes, frame)
