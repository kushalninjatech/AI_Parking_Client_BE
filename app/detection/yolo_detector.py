"""YOLO vehicle detector using ultralytics.
Auto-downloads .pt weights on first run if not found locally.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from app.core.config import settings
from app.core.constants import COCO_VEHICLE_MAP

logger = logging.getLogger(__name__)

YOLO_INPUT_SIZE = settings.YOLO_INPUT_SIZE


class YOLODetector:
    """YOLO detector using ultralytics. Loads once, shared across all cameras."""

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    def load(self) -> bool:
        try:
            from ultralytics import YOLO

            model_path = settings.YOLO_MODEL_PATH

            # Auto-download: if path doesn't exist, ultralytics downloads .pt automatically
            if not os.path.exists(model_path):
                # Extract model name (e.g. "ai_models/yolo11x.pt" -> "yolo11x.pt")
                model_name = os.path.basename(model_path)
                logger.info("Model not found at %s, downloading %s...", model_path, model_name)
                self._model = YOLO(model_name)
                # Save to configured path for next time
                os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
                Path(model_name).rename(model_path)
                logger.info("Model saved to %s", model_path)
            else:
                self._model = YOLO(model_path)

            self._loaded = True
            logger.info(
                "YOLO model loaded: path=%s input_size=%d confidence=%.2f",
                model_path, YOLO_INPUT_SIZE, settings.YOLO_CONFIDENCE,
            )
            return True
        except Exception:
            logger.exception("Failed to load YOLO model")
            return False

    def detect(self, frame: np.ndarray, camera_label: str = "") -> List[Dict]:
        """Run YOLO on full frame. Returns list of vehicle detections.

        Each detection: {
            "class_id": int,
            "confidence": float,
            "bbox": [x1, y1, x2, y2],
            "centroid": [cx, cy],
        }
        """
        if not self._loaded:
            return []

        img_h, img_w = frame.shape[:2]
        t0 = time.monotonic()

        results = self._model(
            frame,
            imgsz=YOLO_INPUT_SIZE,
            conf=settings.YOLO_CONFIDENCE,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in COCO_VEHICLE_MAP:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append({
                "class_id": cls_id,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "centroid": [cx, cy],
            })

        elapsed = time.monotonic() - t0
        logger.info(
            "YOLO detect: %d vehicles in %.2fs (camera=%s frame=%dx%d input=%d)",
            len(detections), elapsed, camera_label, img_w, img_h, YOLO_INPUT_SIZE,
        )
        return detections
