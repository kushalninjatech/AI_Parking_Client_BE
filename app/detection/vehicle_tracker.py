"""Vehicle tracker for multi-capacity zones using centroid matching.

Tracks individual vehicles across detection frames within a zone polygon.
Uses centroid distance matching with entry/exit debounce.

Single-capacity slots (capacity=1) should NOT use this — they use slot-level detection.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.constants import VehicleType

logger = logging.getLogger(__name__)

# Matching threshold as fraction of frame diagonal
MATCH_DISTANCE_FRACTION = 0.15

# Debounce: consecutive frames to confirm entry/exit
ENTRY_CONFIRM_FRAMES = 3
EXIT_CONFIRM_FRAMES = 3

_next_track_id = 0


def _gen_track_id() -> str:
    global _next_track_id
    _next_track_id += 1
    return f"v-{_next_track_id:04d}"


@dataclass
class TrackedVehicle:
    track_id: str
    vehicle_type: VehicleType
    class_id: int
    centroid: Tuple[int, int]
    bbox: List[int]
    confidence: float
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    seen_count: int = 1
    miss_count: int = 0
    confirmed: bool = False


class VehicleTracker:
    """Tracks vehicles inside a zone polygon across frames using centroid matching."""

    def __init__(self, slot_id: int, slot_label: str) -> None:
        self._slot_id = slot_id
        self._slot_label = slot_label
        self._tracked: Dict[str, TrackedVehicle] = {}  # track_id → TrackedVehicle
        self._frame_diagonal: float = 0.0

    def update(
        self,
        detections: List[Dict],
        frame_shape: Tuple[int, int],
    ) -> Dict:
        """Match new detections against tracked vehicles.

        Args:
            detections: List of vehicle detections inside the zone polygon.
                Each: {"class_id": int, "confidence": float, "bbox": [x1,y1,x2,y2],
                       "centroid": [cx,cy], "vehicle_type": VehicleType}
            frame_shape: (height, width) of the frame.

        Returns:
            {"entered": [TrackedVehicle], "exited": [TrackedVehicle],
             "occupied_car": int, "occupied_two_wheeler": int}
        """
        h, w = frame_shape
        self._frame_diagonal = math.sqrt(w * w + h * h)
        max_dist = self._frame_diagonal * MATCH_DISTANCE_FRACTION
        now = time.time()

        # Build list of unmatched detections
        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        # Match existing tracks to new detections (greedy nearest)
        # Sort by distance to get best matches first
        pairs = []
        for track_id, tv in self._tracked.items():
            for i in unmatched_dets:
                det = detections[i]
                dist = self._centroid_distance(tv.centroid, det["centroid"])
                # Also prefer same vehicle type
                type_bonus = 0 if det["vehicle_type"] == tv.vehicle_type else max_dist * 0.3
                pairs.append((dist + type_bonus, track_id, i))

        pairs.sort(key=lambda x: x[0])

        used_dets = set()
        for dist, track_id, det_idx in pairs:
            if track_id in matched_track_ids or det_idx in used_dets:
                continue
            if dist > max_dist:
                break

            # Match found — update tracked vehicle
            det = detections[det_idx]
            tv = self._tracked[track_id]
            tv.centroid = tuple(det["centroid"])
            tv.bbox = det["bbox"]
            tv.confidence = det["confidence"]
            tv.last_seen = now
            tv.miss_count = 0
            tv.seen_count += 1
            if not tv.confirmed and tv.seen_count >= ENTRY_CONFIRM_FRAMES:
                tv.confirmed = True

            matched_track_ids.add(track_id)
            used_dets.add(det_idx)

        # New detections — create pending tracks
        entered = []
        for i in range(len(detections)):
            if i in used_dets:
                continue
            det = detections[i]
            tv = TrackedVehicle(
                track_id=_gen_track_id(),
                vehicle_type=det["vehicle_type"],
                class_id=det["class_id"],
                centroid=tuple(det["centroid"]),
                bbox=det["bbox"],
                confidence=det["confidence"],
                first_seen=now,
                last_seen=now,
            )
            self._tracked[tv.track_id] = tv
            # If debounce disabled, confirm immediately
            if not settings.DETECTION_DEBOUNCE_ENABLED:
                tv.confirmed = True
                tv.seen_count = ENTRY_CONFIRM_FRAMES
            # Entry confirmed after N consecutive frames (checked above on next update)
            if tv.confirmed:
                entered.append(tv)

        # Unmatched tracks — increment miss count
        exited = []
        for track_id, tv in list(self._tracked.items()):
            if track_id in matched_track_ids:
                continue
            tv.miss_count += 1
            exit_threshold = 1 if not settings.DETECTION_DEBOUNCE_ENABLED else EXIT_CONFIRM_FRAMES
            if tv.miss_count >= exit_threshold:
                if tv.confirmed:
                    exited.append(tv)
                del self._tracked[track_id]
            elif not tv.confirmed and tv.miss_count >= ENTRY_CONFIRM_FRAMES:
                # Never confirmed, disappeared — discard silently
                del self._tracked[track_id]

        # Also check newly confirmed entries (seen_count just hit threshold)
        for track_id, tv in self._tracked.items():
            if tv.confirmed and tv.seen_count == ENTRY_CONFIRM_FRAMES and track_id not in matched_track_ids:
                continue
            if tv.confirmed and tv.seen_count == ENTRY_CONFIRM_FRAMES:
                entered.append(tv)

        # Count confirmed vehicles
        occupied_car = sum(
            1 for tv in self._tracked.values()
            if tv.confirmed and tv.vehicle_type == VehicleType.CAR
        )
        occupied_two_wheeler = sum(
            1 for tv in self._tracked.values()
            if tv.confirmed and tv.vehicle_type == VehicleType.TWO_WHEELER
        )

        if entered:
            for tv in entered:
                logger.info(
                    "Zone %s: ENTERED %s track=%s conf=%.2f centroid=(%d,%d)",
                    self._slot_label, tv.vehicle_type.value, tv.track_id,
                    tv.confidence, tv.centroid[0], tv.centroid[1],
                )
        if exited:
            for tv in exited:
                duration = now - tv.first_seen
                logger.info(
                    "Zone %s: EXITED %s track=%s duration=%.0fs",
                    self._slot_label, tv.vehicle_type.value, tv.track_id, duration,
                )

        return {
            "entered": entered,
            "exited": exited,
            "occupied_car": occupied_car,
            "occupied_two_wheeler": occupied_two_wheeler,
        }

    @property
    def confirmed_count(self) -> int:
        return sum(1 for tv in self._tracked.values() if tv.confirmed)

    @property
    def tracked_vehicles(self) -> List[TrackedVehicle]:
        return [tv for tv in self._tracked.values() if tv.confirmed]

    @staticmethod
    def _centroid_distance(c1: Tuple[int, int], c2) -> float:
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
