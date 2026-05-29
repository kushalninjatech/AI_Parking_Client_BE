"""Main detection engine — YOLO + edge/depth anomaly for parking slot classification."""

import json
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.constants import COCO_VEHICLE_MAP, VEHICLE_PRIORITY, SlotState, SlotType, VehicleType
from app.detection.yolo_detector import YOLODetector
from app.detection.depth_estimator import DepthEstimator

logger = logging.getLogger(__name__)

REF_SIZE = (64, 64)

# Laplacian edge-variance ratio: current / calibration.
# Lighting change (smooth gradient) — ratio ≈ 0.8-1.4  (no new edges appear).
# Real object (box, chair)         — ratio ≈ 2.0-8.0  (object edges >> empty floor).
LAP_RATIO_THRESHOLD = 1.8

# Grayscale p90 — secondary confirmation.
GRAY_P90_THRESHOLD = 0.13

# Depth mean-shift — tertiary confirmation when calibration available.
DEPTH_SHIFT_THRESHOLD = 0.03

# Consecutive OBSTRUCTED frames required before reporting.
CONFIRM_FRAMES = 2


class ParkingDetector:
    """Detects slot states using YOLO + edge/depth anomaly."""

    def __init__(self, yolo: YOLODetector, depth: DepthEstimator) -> None:
        self._yolo = yolo
        self._depth = depth
        self._calibrations: Dict[int, Dict] = {}
        self._obstruct_streak: Dict[int, int] = {}

    def set_calibration(self, slot_id: int, calibration_data: bytes) -> None:
        import pickle
        self._calibrations[slot_id] = pickle.loads(calibration_data)

    def calibrate_slot(self, frame: np.ndarray, slot_id: int, polygon: List[List[int]]) -> bytes:
        """Capture empty-slot reference. Returns serialised bytes."""
        import pickle

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._polygon_bbox(polygon)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            logger.error("Slot %d polygon bbox out of frame bounds — cannot calibrate", slot_id)
            return b""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        roi_g = gray[y1:y2, x1:x2]
        ref_g = cv2.resize(roi_g, REF_SIZE)

        # Grayscale brightness-normalised pattern
        gray_mean = float(ref_g.mean())
        ref_g_norm = ref_g / (gray_mean + 1e-8)

        # Laplacian edge-variance of empty slot (lighting-invariant structural reference)
        lap = cv2.Laplacian(ref_g, cv2.CV_32F)
        ref_lap_var = float(lap.var())

        data: Dict = {
            "ref_gray": ref_g_norm.tolist(),
            "ref_gray_mean": gray_mean,
            "ref_lap_var": ref_lap_var,
        }

        depth_map = self._depth.estimate(frame)
        if depth_map is not None:
            roi_d = depth_map[y1:y2, x1:x2].astype(np.float32)
            ref_d = cv2.resize(roi_d, REF_SIZE)
            data["ref_depth"] = ref_d.tolist()
            data["ref_depth_mean"] = float(ref_d.mean())

        logger.info(
            "Slot %d calibrated — gray_mean=%.1f lap_var=%.1f depth_mean=%s",
            slot_id, gray_mean, ref_lap_var,
            f"{data['ref_depth_mean']:.1f}" if "ref_depth_mean" in data else "N/A",
        )
        return pickle.dumps(data)

    def detect_frame(self, frame: np.ndarray, slots: List[Dict], camera_label: str = "") -> List[Dict]:
        """Detect all slot states from a single frame."""
        vehicle_detections = self._yolo.detect(frame, camera_label=camera_label)
        logger.debug("YOLO: %d detections", len(vehicle_detections))

        depth_map = self._depth.estimate(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        results = []
        for slot in slots:
            polygon = (
                json.loads(slot["polygon_coords"])
                if isinstance(slot["polygon_coords"], str)
                else slot["polygon_coords"]
            )
            slot_type = SlotType(slot.get("slot_type", SlotType.GENERAL.value))
            base = {"id": slot["id"], "label": slot["label"], "slot_type": slot_type}

            if not polygon:
                results.append({**base, "state": SlotState.EMPTY, "confidence": 0.0,
                                "detected_vehicle_type": None, "is_mismatched": False})
                continue

            # Collect all vehicle detections whose centroids fall inside this polygon
            polygon_np = np.array(polygon, dtype=np.int32)
            matched_vehicles = []
            for det in vehicle_detections:
                cx, cy = det["centroid"]
                if cv2.pointPolygonTest(polygon_np, (float(cx), float(cy)), False) >= 0:
                    vtype = COCO_VEHICLE_MAP.get(det["class_id"])
                    logger.info(
                        "Slot %s: matched class_id=%d (%s) conf=%.2f centroid=(%d,%d)",
                        slot["label"], det["class_id"], vtype, det["confidence"], cx, cy,
                    )
                    if vtype:
                        matched_vehicles.append((vtype, det["confidence"]))

            if matched_vehicles:
                # Largest vehicle wins (highest priority)
                best_type, best_conf = max(matched_vehicles, key=lambda x: VEHICLE_PRIORITY.get(x[0], 0))
                is_mismatched = slot_type != SlotType.GENERAL and best_type.value != slot_type.value
                self._obstruct_streak[slot["id"]] = 0
                results.append({**base, "state": SlotState.VEHICLE, "confidence": best_conf,
                                "detected_vehicle_type": best_type, "is_mismatched": is_mismatched})
                continue

            if slot["id"] in self._calibrations:
                is_obstructed, confidence = self._check_anomaly(gray, depth_map, polygon, slot["id"])
                sid = slot["id"]
                if is_obstructed:
                    self._obstruct_streak[sid] = self._obstruct_streak.get(sid, 0) + 1
                else:
                    self._obstruct_streak[sid] = 0
                confirmed = self._obstruct_streak.get(sid, 0) >= CONFIRM_FRAMES
                state = SlotState.OBSTRUCTED if confirmed else SlotState.EMPTY
                results.append({**base, "state": state, "confidence": confidence if confirmed else 0.0,
                                "detected_vehicle_type": None, "is_mismatched": False})
            else:
                results.append({**base, "state": SlotState.EMPTY, "confidence": 0.0,
                                "detected_vehicle_type": None, "is_mismatched": False})

        return results

    def _check_anomaly(
        self,
        gray: np.ndarray,
        depth_map: Optional[np.ndarray],
        polygon: List[List[int]],
        slot_id: int,
    ) -> Tuple[bool, float]:
        cal = self._calibrations.get(slot_id)
        if not cal or "ref_gray" not in cal:
            logger.warning("Slot %d needs recalibration — treating as EMPTY", slot_id)
            return False, 0.0

        h, w = gray.shape[:2]
        x1, y1, x2, y2 = self._polygon_bbox(polygon)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return False, 0.0

        roi_g = gray[y1:y2, x1:x2]
        cur_g = cv2.resize(roi_g, REF_SIZE)

        # --- Laplacian edge-variance ratio (lighting-invariant) ---
        cur_lap = cv2.Laplacian(cur_g, cv2.CV_32F)
        cur_lap_var = float(cur_lap.var())
        ref_lap_var = cal.get("ref_lap_var", 1.0)
        lap_ratio = cur_lap_var / (ref_lap_var + 1.0)
        lap_triggered = lap_ratio > LAP_RATIO_THRESHOLD

        # --- Grayscale p90 diff ---
        cur_g_norm = cur_g / (cur_g.mean() + 1e-8)
        ref_g_norm = np.array(cal["ref_gray"], dtype=np.float32)
        if ref_g_norm.shape != cur_g_norm.shape:
            ref_g_norm = cv2.resize(ref_g_norm, REF_SIZE)
        gray_p90 = float(np.percentile(np.abs(cur_g_norm - ref_g_norm), 90))
        gray_triggered = gray_p90 > GRAY_P90_THRESHOLD

        # --- Depth shift ---
        depth_shift = 0.0
        if depth_map is not None and "ref_depth_mean" in cal:
            roi_d = depth_map[y1:y2, x1:x2].astype(np.float32)
            cur_d = cv2.resize(roi_d, REF_SIZE)
            depth_shift = (cur_d.mean() - cal["ref_depth_mean"]) / 255.0
        depth_triggered = depth_shift > DEPTH_SHIFT_THRESHOLD

        # 2-of-3 majority vote: lap + gray + depth.
        # Degrades gracefully if depth is unavailable (lap+gray still wins).
        triggered_count = sum([lap_triggered, gray_triggered, depth_triggered])
        is_obstructed = triggered_count >= 2

        lap_conf = min(lap_ratio / LAP_RATIO_THRESHOLD / 2.5, 1.0)
        if depth_map is not None and "ref_depth_mean" in cal and depth_shift > 0:
            depth_conf = min(depth_shift / DEPTH_SHIFT_THRESHOLD / 2.5, 1.0)
            confidence = min((lap_conf + depth_conf) / 2, 1.0)
        else:
            confidence = lap_conf

        logger.info(
            "Slot %d lap_ratio=%.2f gray_p90=%.3f depth_shift=%.3f obstructed=%s",
            slot_id, lap_ratio, gray_p90, depth_shift, is_obstructed,
        )
        return is_obstructed, confidence

    @staticmethod
    def _polygon_bbox(polygon: List[List[int]]) -> Tuple[int, int, int, int]:
        pts = np.array(polygon)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        return int(x1), int(y1), int(x2), int(y2)
