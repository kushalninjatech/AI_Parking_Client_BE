"""Main detection engine — YOLO + edge/depth anomaly for parking slot classification."""

import json
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.constants import COCO_VEHICLE_MAP, VEHICLE_PRIORITY, SlotState, SlotType, VehicleType
from app.detection.yolo_detector import YOLODetector
from app.detection.depth_estimator import DepthEstimator
from app.detection.vehicle_tracker import VehicleTracker

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
    """Detects slot states using YOLO + edge/depth anomaly OR Gemini LLM vision.

    Backend is selected via DETECTION_BACKEND env var ("yolo" or "gemini").
    """

    def __init__(self, yolo: YOLODetector, depth: DepthEstimator, gemini=None) -> None:
        self._yolo = yolo
        self._depth = depth
        self._gemini = gemini  # GeminiDetector instance (optional)
        self._backend = settings.DETECTION_BACKEND.lower()
        self._calibrations: Dict[int, Dict] = {}
        self._last_detections: List[Dict] = []
        self._obstruct_streak: Dict[int, int] = {}
        self._trackers: Dict[int, VehicleTracker] = {}  # slot_id → VehicleTracker
        self._last_vehicle_events: List[Dict] = []  # entry/exit events from last frame

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

    def generate_debug_frame(self, frame: np.ndarray, vehicle_detections: List[Dict], slots: List[Dict], results: List[Dict]) -> np.ndarray:
        """Draw YOLO bboxes + slot polygons on frame for debugging."""
        debug = frame.copy()
        # Draw all vehicle bboxes in red
        for det in vehicle_detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["centroid"]
            cls = det["class_id"]
            conf = det["confidence"]
            color = (0, 0, 255) if cls == 2 else (255, 100, 0)  # red=car, orange=2w
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
            cv2.circle(debug, (cx, cy), 5, color, -1)
            label = f"{'CAR' if cls == 2 else '2W'} {conf:.0%}"
            cv2.putText(debug, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Draw slot polygons with state colors
        for slot, res in zip(slots, results):
            polygon = slot["polygon_coords"]
            if isinstance(polygon, str):
                polygon = json.loads(polygon)
            if not polygon:
                continue
            pts = np.array(polygon, dtype=np.int32)
            state = res.get("state", "EMPTY")
            sc = (0, 255, 0) if state == "EMPTY" else (0, 0, 255) if state == "VEHICLE" else (0, 165, 255)
            cv2.polylines(debug, [pts], True, sc, 3)
            cx_p, cy_p = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            vtype = res.get("detected_vehicle_type")
            vtype_str = vtype.value if hasattr(vtype, "value") else (vtype or "")
            text = f"{slot['label']}: {state}"
            if vtype_str:
                text += f" ({vtype_str})"
            occ_c = res.get("occupied_car", 0)
            occ_2w = res.get("occupied_two_wheeler", 0)
            if occ_c or occ_2w:
                text += f" C:{occ_c} 2W:{occ_2w}"
            cv2.putText(debug, text, (cx_p - 60, cy_p), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)
        return debug

    def detect_frame(self, frame: np.ndarray, slots: List[Dict], camera_label: str = "") -> List[Dict]:
        """Detect all slot states from a single frame."""
        # Route to Gemini backend if configured and loaded
        if self._backend == "gemini" and self._gemini and self._gemini.is_loaded:
            return self._detect_frame_gemini(frame, slots, camera_label)

        return self._detect_frame_yolo(frame, slots, camera_label)

    def _detect_frame_gemini(self, frame: np.ndarray, slots: List[Dict], camera_label: str = "") -> List[Dict]:
        """Gemini LLM vision backend — sends cropped ROI per slot."""
        self._last_detections = []
        self._last_vehicle_events = []

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
                                "detected_vehicle_type": None, "is_mismatched": False,
                                "occupied_car": 0, "occupied_two_wheeler": 0})
                continue

            # Send cropped ROI to Gemini
            gemini_result = self._gemini.detect_slot(frame, polygon, camera_label)

            # Map Gemini counts to our format
            # Only car and two_wheeler are valid vehicles; everything else is obstruction
            occ_car = gemini_result["car"]
            occ_2w = gemini_result["two_wheeler"]
            total_vehicles = occ_car + occ_2w
            other_objects = (gemini_result["auto_rickshaw"] + gemini_result["bus"]
                            + gemini_result["truck"] + gemini_result["tempo"])
            is_obstructed = gemini_result["is_obstructed"] or other_objects > 0
            confidence = gemini_result["confidence"]

            # Determine state
            if total_vehicles > 0:
                state = SlotState.VEHICLE
            elif is_obstructed:
                state = SlotState.OBSTRUCTED
            else:
                state = SlotState.EMPTY

            # Determine dominant vehicle type
            if occ_car > 0 and occ_car >= occ_2w:
                detected_vtype = VehicleType.CAR
            elif occ_2w > 0:
                detected_vtype = VehicleType.TWO_WHEELER
            else:
                detected_vtype = None

            is_mismatched = (
                slot_type != SlotType.GENERAL
                and detected_vtype is not None
                and detected_vtype.value != slot_type.value
            )

            results.append({
                **base,
                "state": state,
                "confidence": confidence,
                "detected_vehicle_type": detected_vtype,
                "is_mismatched": is_mismatched,
                "occupied_car": occ_car,
                "occupied_two_wheeler": occ_2w,
            })

        return results

    def _detect_frame_yolo(self, frame: np.ndarray, slots: List[Dict], camera_label: str = "") -> List[Dict]:
        """YOLO + Depth backend — original on-device detection."""
        vehicle_detections = self._yolo.detect(frame, camera_label=camera_label)
        self._last_detections = vehicle_detections
        self._last_vehicle_events = []
        logger.debug("YOLO: %d detections", len(vehicle_detections))

        depth_map = self._depth.estimate(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        img_h, img_w = frame.shape[:2]

        results = []
        for slot in slots:
            polygon = (
                json.loads(slot["polygon_coords"])
                if isinstance(slot["polygon_coords"], str)
                else slot["polygon_coords"]
            )
            slot_type = SlotType(slot.get("slot_type", SlotType.GENERAL.value))
            base = {"id": slot["id"], "label": slot["label"], "slot_type": slot_type}

            capacity_car = slot.get("capacity_car", 0)
            capacity_two_wheeler = slot.get("capacity_two_wheeler", 0)
            total_capacity = capacity_car + capacity_two_wheeler

            if not polygon:
                results.append({**base, "state": SlotState.EMPTY, "confidence": 0.0,
                                "detected_vehicle_type": None, "is_mismatched": False,
                                "occupied_car": 0, "occupied_two_wheeler": 0})
                continue

            # Collect all vehicle detections whose centroids fall inside this polygon
            polygon_np = np.array(polygon, dtype=np.int32)
            px_min, py_min = polygon_np.min(axis=0)
            px_max, py_max = polygon_np.max(axis=0)
            logger.info(
                "Slot %s: polygon=(%d,%d)-(%d,%d), %d vehicles to check",
                slot["label"], px_min, py_min, px_max, py_max, len(vehicle_detections),
            )

            matched_detections = []
            for det in vehicle_detections:
                cx, cy = det["centroid"]
                if cv2.pointPolygonTest(polygon_np, (float(cx), float(cy)), False) >= 0:
                    vtype = COCO_VEHICLE_MAP.get(det["class_id"])
                    if vtype:
                        matched_detections.append({**det, "vehicle_type": vtype})

            # ── Multi-capacity zone → vehicle tracker ──
            if total_capacity > 1:
                result = self._detect_multi_capacity(
                    slot, base, matched_detections, (img_h, img_w),
                )
                results.append(result)
                continue

            # ── Single-capacity slot → slot-level detection ──
            matched_vehicles = [(d["vehicle_type"], d["confidence"]) for d in matched_detections]

            if matched_vehicles:
                best_type, best_conf = max(matched_vehicles, key=lambda x: VEHICLE_PRIORITY.get(x[0], 0))
                is_mismatched = slot_type != SlotType.GENERAL and best_type.value != slot_type.value
                self._obstruct_streak[slot["id"]] = 0
                results.append({**base, "state": SlotState.VEHICLE, "confidence": best_conf,
                                "detected_vehicle_type": best_type, "is_mismatched": is_mismatched,
                                "occupied_car": sum(1 for v, _ in matched_vehicles if v == VehicleType.CAR),
                                "occupied_two_wheeler": sum(1 for v, _ in matched_vehicles if v == VehicleType.TWO_WHEELER)})
                continue

            # Anomaly/depth fallback for single-capacity slots
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
                                "detected_vehicle_type": None, "is_mismatched": False,
                                "occupied_car": 0, "occupied_two_wheeler": 0})
            else:
                results.append({**base, "state": SlotState.EMPTY, "confidence": 0.0,
                                "detected_vehicle_type": None, "is_mismatched": False,
                                "occupied_car": 0, "occupied_two_wheeler": 0})

        return results

    def _detect_multi_capacity(
        self, slot: Dict, base: Dict, matched_detections: List[Dict],
        frame_shape: Tuple[int, int],
    ) -> Dict:
        """Multi-capacity zone detection.

        If TRACKER_ENTRY_FRAMES=1: immediate counts from raw YOLO, no tracker.
        If TRACKER_ENTRY_FRAMES>1: tracker with debounce for stable entry/exit.
        """
        slot_id = slot["id"]
        use_tracker = settings.TRACKER_ENTRY_FRAMES > 1

        # --- Raw YOLO counts (always computed) ---
        raw_occ_car = sum(1 for d in matched_detections if d["vehicle_type"] == VehicleType.CAR)
        raw_occ_2w = sum(1 for d in matched_detections if d["vehicle_type"] == VehicleType.TWO_WHEELER)

        if use_tracker:
            # --- Tracker mode: debounced entry/exit + counts from confirmed tracks ---
            if slot_id not in self._trackers:
                self._trackers[slot_id] = VehicleTracker(slot_id, slot["label"])

            tracker = self._trackers[slot_id]
            tracker_result = tracker.update(matched_detections, frame_shape)

            for tv in tracker_result["entered"]:
                self._last_vehicle_events.append({
                    "event_type": "ENTERED",
                    "slot_label": slot["label"],
                    "slot_id": slot_id,
                    "vehicle_type": tv.vehicle_type.value,
                    "track_id": tv.track_id,
                    "confidence": tv.confidence,
                    "centroid": list(tv.centroid),
                    "bbox": tv.bbox,
                })
            for tv in tracker_result["exited"]:
                self._last_vehicle_events.append({
                    "event_type": "EXITED",
                    "slot_label": slot["label"],
                    "slot_id": slot_id,
                    "vehicle_type": tv.vehicle_type.value,
                    "track_id": tv.track_id,
                    "duration_seconds": round(tv.last_seen - tv.first_seen),
                    "centroid": list(tv.centroid),
                    "bbox": tv.bbox,
                })

            occ_car = tracker_result["occupied_car"]
            occ_2w = tracker_result["occupied_two_wheeler"]
        else:
            # --- Immediate mode: raw YOLO counts, every detection = entry event ---
            occ_car = raw_occ_car
            occ_2w = raw_occ_2w

            for d in matched_detections:
                self._last_vehicle_events.append({
                    "event_type": "ENTERED",
                    "slot_label": slot["label"],
                    "slot_id": slot_id,
                    "vehicle_type": d["vehicle_type"].value,
                    "track_id": f"raw-{slot_id}-{d['class_id']}-{d['centroid'][0]}",
                    "confidence": d["confidence"],
                    "centroid": list(d["centroid"]),
                    "bbox": d["bbox"],
                })

        total_occ = occ_car + occ_2w
        state = SlotState.VEHICLE if total_occ > 0 else SlotState.EMPTY
        best_conf = max((d["confidence"] for d in matched_detections), default=0.0)
        best_type = None
        if matched_detections:
            best_det = max(matched_detections, key=lambda d: VEHICLE_PRIORITY.get(d["vehicle_type"], 0))
            best_type = best_det["vehicle_type"]

        return {
            **base, "state": state, "confidence": best_conf,
            "detected_vehicle_type": best_type, "is_mismatched": False,
            "occupied_car": occ_car, "occupied_two_wheeler": occ_2w,
        }

    @property
    def last_vehicle_events(self) -> List[Dict]:
        """Entry/exit events from the last detect_frame call."""
        return self._last_vehicle_events

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
