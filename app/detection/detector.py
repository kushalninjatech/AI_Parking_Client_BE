"""Main detection engine. Processes full frame through YOLO + depth,
classifies each slot as VEHICLE / EMPTY / OBSTRUCTED.

Ported from R&D/poc — combines app_v2.py detection with parking_detector.py grid logic.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.constants import SlotState
from app.detection.yolo_detector import YOLODetector
from app.detection.depth_estimator import DepthEstimator

logger = logging.getLogger(__name__)

# Depth anomaly detection constants (from POC)
GRID_ROWS = 6
GRID_COLS = 6
CELL_THRESHOLD = 0.35
MIN_TRIGGERED_RATIO = 0.05
HOT_CELL_THRESHOLD = 0.8


class ParkingDetector:
    """Detects slot states using YOLO + depth estimation on full frame."""

    def __init__(self, yolo: YOLODetector, depth: DepthEstimator) -> None:
        self._yolo = yolo
        self._depth = depth
        # Per-slot calibration: {slot_id: {"norm": ndarray, "ref_grads": list}}
        self._calibrations: Dict[int, Dict] = {}

    def set_calibration(self, slot_id: int, calibration_data: bytes) -> None:
        """Load calibration data for a slot (from DB)."""
        import pickle
        self._calibrations[slot_id] = pickle.loads(calibration_data)

    def calibrate_slot(self, frame: np.ndarray, slot_id: int, polygon: List[List[int]]) -> bytes:
        """Calibrate a slot with an empty reference frame. Returns serialized data."""
        import pickle

        depth_map = self._depth.estimate(frame)
        if depth_map is None:
            return b""

        # Extract ROI from depth map using polygon bounding box
        x1, y1, x2, y2 = self._polygon_bbox(polygon)
        roi_depth = depth_map[y1:y2, x1:x2].astype(np.float32)

        # Z-score normalize
        mean, std = roi_depth.mean(), roi_depth.std() + 1e-8
        norm = (roi_depth - mean) / std

        # Grid cell gradient energies
        h, w = norm.shape
        ref_grads = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                cy1 = r * h // GRID_ROWS
                cy2 = (r + 1) * h // GRID_ROWS
                cx1 = c * w // GRID_COLS
                cx2 = (c + 1) * w // GRID_COLS
                cell = norm[cy1:cy2, cx1:cx2]
                gx = cv2.Sobel(cell, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(cell, cv2.CV_32F, 0, 1, ksize=3)
                grad = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
                ref_grads.append(grad)

        data = {"norm_mean": float(mean), "norm_std": float(std), "ref_grads": ref_grads}
        return pickle.dumps(data)

    def detect_frame(
        self, frame: np.ndarray, slots: List[Dict]
    ) -> List[Dict]:
        """Detect all slot states from a single frame.

        Args:
            frame: Full camera frame (BGR)
            slots: List of {id, label, polygon_coords (JSON string)}

        Returns:
            List of {id, label, state, confidence}
        """
        # Step 1: YOLO on full frame
        vehicle_detections = self._yolo.detect(frame)
        logger.debug("YOLO: %d vehicles detected", len(vehicle_detections))

        # Step 2: Depth on full frame
        depth_map = self._depth.estimate(frame)

        results = []
        for slot in slots:
            polygon = json.loads(slot["polygon_coords"]) if isinstance(slot["polygon_coords"], str) else slot["polygon_coords"]
            if not polygon:
                results.append({"id": slot["id"], "label": slot["label"], "state": SlotState.EMPTY, "confidence": 0.0})
                continue

            # Step 3: Check if any vehicle centroid is inside this slot's polygon
            polygon_np = np.array(polygon, dtype=np.int32)
            has_vehicle = False
            vehicle_conf = 0.0

            for det in vehicle_detections:
                cx, cy = det["centroid"]
                inside = cv2.pointPolygonTest(polygon_np, (float(cx), float(cy)), False)
                if inside >= 0:
                    has_vehicle = True
                    vehicle_conf = det["confidence"]
                    break

            if has_vehicle:
                results.append({"id": slot["id"], "label": slot["label"], "state": SlotState.VEHICLE, "confidence": vehicle_conf})
                continue

            # Step 4: No vehicle — check depth for obstruction
            if depth_map is not None and slot["id"] in self._calibrations:
                is_obstructed, confidence = self._check_depth_anomaly(
                    depth_map, polygon, slot["id"]
                )
                state = SlotState.OBSTRUCTED if is_obstructed else SlotState.EMPTY
                results.append({"id": slot["id"], "label": slot["label"], "state": state, "confidence": confidence})
            else:
                # No depth or no calibration — assume empty
                results.append({"id": slot["id"], "label": slot["label"], "state": SlotState.EMPTY, "confidence": 0.0})

        return results

    def _check_depth_anomaly(
        self, depth_map: np.ndarray, polygon: List[List[int]], slot_id: int
    ) -> Tuple[bool, float]:
        """Compare depth ROI against calibrated reference. Returns (is_obstructed, confidence)."""
        cal = self._calibrations.get(slot_id)
        if not cal:
            return False, 0.0

        x1, y1, x2, y2 = self._polygon_bbox(polygon)
        roi_depth = depth_map[y1:y2, x1:x2].astype(np.float32)

        # Z-score normalize
        mean, std = roi_depth.mean(), roi_depth.std() + 1e-8
        norm = (roi_depth - mean) / std

        # Grid scoring
        h, w = norm.shape
        ref_grads = cal["ref_grads"]
        triggered = 0
        max_score = 0.0

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                cy1 = r * h // GRID_ROWS
                cy2 = (r + 1) * h // GRID_ROWS
                cx1 = c * w // GRID_COLS
                cx2 = (c + 1) * w // GRID_COLS
                cell = norm[cy1:cy2, cx1:cx2]

                # Mean depth diff
                ref_norm_mean = 0.0  # calibrated reference is normalized to ~0
                mean_diff = abs(float(cell.mean()) - ref_norm_mean)

                # Gradient excess
                gx = cv2.Sobel(cell, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(cell, cv2.CV_32F, 0, 1, ksize=3)
                grad = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
                idx = r * GRID_COLS + c
                ref_grad = ref_grads[idx] if idx < len(ref_grads) else 0.01
                grad_excess = max(0.0, grad / (ref_grad + 1e-4) - 1.0)

                score = mean_diff + 0.3 * grad_excess
                max_score = max(max_score, score)
                if score > CELL_THRESHOLD:
                    triggered += 1

        total_cells = GRID_ROWS * GRID_COLS
        ratio = triggered / total_cells
        is_obstructed = ratio >= MIN_TRIGGERED_RATIO or max_score > HOT_CELL_THRESHOLD
        confidence = min((ratio / 0.25) * 0.5 + (max_score / 2.0) * 0.5, 1.0)

        return is_obstructed, confidence

    @staticmethod
    def _polygon_bbox(polygon: List[List[int]]) -> Tuple[int, int, int, int]:
        """Get bounding box from polygon points."""
        pts = np.array(polygon)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        return int(x1), int(y1), int(x2), int(y2)
