"""Detection loop — runs sequentially for all cameras at configured interval."""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional

from app.camera.factory import create_camera
from app.core.config import settings
from app.core.constants import SlotState
from app.detection.detector import ParkingDetector

logger = logging.getLogger(__name__)


class DetectionLoop:
    """Main detection loop. Processes all cameras sequentially."""

    def __init__(self, detector: ParkingDetector) -> None:
        self._detector = detector
        self._camera_configs: Dict[int, Dict] = {}  # camera_id → {source, slots, ...}
        self._running = False
        self._slot_states: Dict[int, SlotState] = {}  # slot_id → last known state
        self._on_state_change = None  # callback
        self._on_frame_captured = None  # callback for latest frame

    def set_callbacks(
        self,
        on_state_change=None,
        on_frame_captured=None,
    ) -> None:
        self._on_state_change = on_state_change
        self._on_frame_captured = on_frame_captured

    def add_camera(self, camera_id: int, source: str, slots: List[Dict]) -> None:
        """Register a camera with its slots for detection."""
        self._camera_configs[camera_id] = {"source": source, "slots": slots}
        logger.info("Camera %d registered with %d slots", camera_id, len(slots))

    def remove_camera(self, camera_id: int) -> None:
        """Remove a camera from the detection loop."""
        self._camera_configs.pop(camera_id, None)
        logger.info("Camera %d removed", camera_id)

    def update_slots(self, camera_id: int, slots: List[Dict]) -> None:
        """Update slot config for a camera (after ROI change)."""
        if camera_id in self._camera_configs:
            self._camera_configs[camera_id]["slots"] = slots

    async def run(self) -> None:
        """Main loop — runs forever until stopped."""
        self._running = True
        logger.info("Detection loop started (interval=%ds)", settings.DETECTION_INTERVAL)

        while self._running:
            start = time.time()

            for camera_id, config in list(self._camera_configs.items()):
                try:
                    await self._process_camera(camera_id, config)
                except Exception:
                    logger.exception("Error processing camera %d", camera_id)

            elapsed = time.time() - start
            sleep_time = max(0, settings.DETECTION_INTERVAL - elapsed)
            logger.debug("Detection cycle took %.1fs, sleeping %.1fs", elapsed, sleep_time)
            await asyncio.sleep(sleep_time)

    def stop(self) -> None:
        """Stop the detection loop."""
        self._running = False
        logger.info("Detection loop stopped")

    async def _process_camera(self, camera_id: int, config: Dict) -> None:
        """Process a single camera: capture → detect → classify → callback."""
        source = config["source"]
        slots = config["slots"]

        if not slots:
            return

        # Open camera, capture, release immediately
        cam = create_camera(source)
        if not cam.open():
            logger.error("Failed to open camera %d (%s)", camera_id, source)
            return

        ok, frame = cam.read()
        cam.release()

        if not ok or frame is None:
            logger.warning("Failed to capture from camera %d", camera_id)
            return

        # Notify frame captured (for snapshot/streaming)
        if self._on_frame_captured:
            self._on_frame_captured(camera_id, frame)

        # Run detection
        slot_dicts = [
            {"id": s["id"], "label": s["label"], "polygon_coords": s["polygon_coords"]}
            for s in slots if s.get("polygon_coords")
        ]

        if not slot_dicts:
            return

        results = self._detector.detect_frame(frame, slot_dicts)

        # Check for state changes
        changes = []
        for r in results:
            slot_id = r["id"]
            new_state = r["state"]
            old_state = self._slot_states.get(slot_id)

            if old_state != new_state:
                self._slot_states[slot_id] = new_state
                changes.append(r)
                logger.info("Slot %s: %s → %s (conf=%.2f)", r["label"], old_state, new_state, r["confidence"])

        # Callback with all results + changes
        if self._on_state_change and results:
            self._on_state_change(camera_id, results, changes)
