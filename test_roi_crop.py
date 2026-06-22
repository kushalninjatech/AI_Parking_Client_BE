"""Test script: verify debug frame is cropped to slot polygon ROI.

Captures a frame from each active camera, generates the debug frame,
crops to the union slot ROI, and saves both full + cropped images
to /tmp/roi_test/ for visual comparison.

Usage:
    cd AI_Parking_Client_BE
    python test_roi_crop.py
"""

import json
import sys
import os

import cv2
import numpy as np

# Ensure app imports work
sys.path.insert(0, os.path.dirname(__file__))

from app.db.session import SessionLocal
from app.models.camera import Camera
from app.models.parking_slot import ParkingSlot  # noqa: F401 — needed by Camera relationship
from app.detection.detection_loop import DetectionLoop


OUT_DIR = "/tmp/roi_test"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    db = SessionLocal()
    try:
        cameras = db.query(Camera).filter(Camera.is_active == True).all()
        if not cameras:
            print("No active cameras found in DB.")
            return

        for cam in cameras:
            slot_dicts = [
                {"id": s.id, "label": s.label, "polygon_coords": s.polygon_coords,
                 "slot_type": s.slot_type or "GENERAL"}
                for s in cam.slots if s.polygon_coords
            ]
            if not slot_dicts:
                print(f"[{cam.label}] No slots with polygons, skipping.")
                continue

            # Capture frame
            print(f"[{cam.label}] Opening {cam.source} ...")
            cap = cv2.VideoCapture(cam.source)
            if not cap.isOpened():
                print(f"[{cam.label}] Failed to open camera.")
                continue
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                print(f"[{cam.label}] Failed to capture frame.")
                continue

            h, w = frame.shape[:2]
            print(f"[{cam.label}] Frame: {w}x{h}, Slots: {len(slot_dicts)}")

            # Draw slot polygons on full frame
            full_debug = frame.copy()
            for slot in slot_dicts:
                polygon = slot["polygon_coords"]
                if isinstance(polygon, str):
                    polygon = json.loads(polygon)
                if not polygon:
                    continue
                pts = np.array(polygon, dtype=np.int32)
                cv2.polylines(full_debug, [pts], True, (0, 0, 255), 3)
                cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                cv2.putText(full_debug, slot["label"], (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Crop using the same method as detection_loop
            cropped = DetectionLoop._crop_to_slots_roi(full_debug, slot_dicts)
            ch, cw = cropped.shape[:2]
            pct = (cw * ch) / (w * h) * 100

            # Save
            safe_label = cam.label.replace(" ", "_").replace("/", "_")
            full_path = os.path.join(OUT_DIR, f"{safe_label}_full.jpg")
            crop_path = os.path.join(OUT_DIR, f"{safe_label}_cropped.jpg")
            cv2.imwrite(full_path, full_debug)
            cv2.imwrite(crop_path, cropped)

            print(f"[{cam.label}] Cropped: {cw}x{ch} ({pct:.0f}% of frame)")
            print(f"  Full:    {full_path}")
            print(f"  Cropped: {crop_path}")

    finally:
        db.close()

    print(f"\nAll images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
