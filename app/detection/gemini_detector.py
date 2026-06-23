"""Gemini LLM vision detector — sends cropped ROI images for vehicle counting + obstruction detection.

Replaces YOLO + Depth pipeline. Same output contract as ParkingDetector.detect_frame().
"""

import base64
import json
import logging
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.constants import SlotState, SlotType, VehicleType

logger = logging.getLogger(__name__)

PROMPT = """This is a cropped image of a parking zone from a CCTV camera. Count ONLY the vehicles that are parked/stationary WITHIN this cropped area.

Return ONLY valid JSON (no markdown, no code blocks):
{
  "car": 0,
  "two_wheeler": 0,
  "auto_rickshaw": 0,
  "bus": 0,
  "truck": 0,
  "tempo": 0,
  "is_obstructed": false,
  "obstruction_type": null,
  "confidence": 0.9
}

Rules:
- Count ONLY vehicles visible in this cropped image. Do NOT guess or infer vehicles outside the frame.
- Count only parked/stationary vehicles, ignore moving ones or pedestrians.
- "car" includes sedans, SUVs, hatchbacks, jeeps. Even partially visible cars at edges count as 1.
- "two_wheeler" includes motorcycles, scooters, mopeds. Even partially visible ones count as 1.
- "auto_rickshaw" includes 3-wheeled auto rickshaws.
- "tempo" includes small commercial goods vehicles, mini trucks.
- "is_obstructed": true if any non-vehicle object is blocking parking spots in this area.
- Obstructions: street vendor carts, food stalls, construction material, debris, barricades, fallen objects, encroachments — anything that is NOT a parked vehicle but occupies parking space.
- "obstruction_type": short description (e.g. "street_vendor_cart", "construction_material") or null if none.
- "confidence": your confidence in the count accuracy (0.0 to 1.0). Set below 0.5 if image is dark/blurry."""


class GeminiDetector:
    """Detects vehicles and obstructions using Gemini LLM vision API."""

    def __init__(self) -> None:
        self._client = None
        self._model_name = settings.GEMINI_MODEL
        self._loaded = False

    def load(self) -> bool:
        """Initialize the Gemini client."""
        if not settings.GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not set — cannot use Gemini backend")
            return False

        try:
            from google import genai
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self._loaded = True
            logger.info("Gemini detector loaded: model=%s", self._model_name)
            return True
        except Exception:
            logger.exception("Failed to initialize Gemini client")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def detect_slot(self, frame: np.ndarray, polygon: List[List[int]], camera_label: str = "") -> Dict:
        """Send cropped ROI to Gemini and parse response.

        Args:
            frame: full BGR frame
            polygon: [[x1,y1], [x2,y2], ...] slot polygon
            camera_label: for logging

        Returns:
            dict with vehicle counts + obstruction info
        """
        if not self._loaded:
            logger.warning("Gemini not loaded — returning empty result")
            return self._empty_result()

        # Crop ROI from polygon bounding box
        roi = self._crop_polygon_roi(frame, polygon)
        if roi is None:
            return self._empty_result()

        frame_h, frame_w = frame.shape[:2]
        roi_h, roi_w = roi.shape[:2]
        logger.info(
            "Gemini ROI crop [%s]: frame=%dx%d → roi=%dx%d (%.0f%% of frame)",
            camera_label, frame_w, frame_h, roi_w, roi_h,
            (roi_w * roi_h) / (frame_w * frame_h) * 100,
        )

        # Encode to JPEG bytes
        _, jpeg_bytes = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 85])

        try:
            from google.genai import types

            t0 = time.monotonic()
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[
                    types.Part.from_bytes(data=jpeg_bytes.tobytes(), mime_type="image/jpeg"),
                    PROMPT,
                ],
            )
            elapsed = time.monotonic() - t0

            result = self._parse_response(response.text)
            logger.info(
                "Gemini detect [%s]: car=%d 2w=%d auto=%d bus=%d truck=%d tempo=%d obstructed=%s (%.2fs)",
                camera_label, result["car"], result["two_wheeler"],
                result["auto_rickshaw"], result["bus"], result["truck"], result["tempo"],
                result["is_obstructed"], elapsed,
            )
            return result

        except Exception:
            logger.exception("Gemini API call failed for %s", camera_label)
            return self._empty_result()

    def _crop_polygon_roi(self, frame: np.ndarray, polygon: List[List[int]]) -> Optional[np.ndarray]:
        """Crop frame to polygon shape — black outside polygon, then bbox crop."""
        h, w = frame.shape[:2]
        pts = np.array(polygon, dtype=np.int32)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Mask outside polygon to black
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        return masked[y1:y2, x1:x2]

    def _parse_response(self, text: str) -> Dict:
        """Parse Gemini JSON response with fallback handling."""
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (``` markers)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini response: %s", text[:200])
            return self._empty_result()

        return {
            "car": int(data.get("car", 0)),
            "two_wheeler": int(data.get("two_wheeler", 0)),
            "auto_rickshaw": int(data.get("auto_rickshaw", 0)),
            "bus": int(data.get("bus", 0)),
            "truck": int(data.get("truck", 0)),
            "tempo": int(data.get("tempo", 0)),
            "is_obstructed": bool(data.get("is_obstructed", False)),
            "obstruction_type": data.get("obstruction_type"),
            "confidence": float(data.get("confidence", 0.5)),
        }

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "car": 0, "two_wheeler": 0, "auto_rickshaw": 0,
            "bus": 0, "truck": 0, "tempo": 0,
            "is_obstructed": False, "obstruction_type": None,
            "confidence": 0.0,
        }
