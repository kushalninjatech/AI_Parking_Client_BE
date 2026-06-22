"""Test: capture frame, crop to each slot's polygon bbox, save to debug/.

Usage:
    cd AI_Parking_Client_BE
    python test_roi_crop.py
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from app.db.session import SessionLocal
from app.models.camera import Camera
from app.models.parking_slot import ParkingSlot  # noqa: F401

OUT_DIR = os.path.join(os.path.dirname(__file__), "debug")
os.makedirs(OUT_DIR, exist_ok=True)


def crop_polygon(frame, polygon):
    """Crop frame to only the area inside the polygon (black outside)."""
    pts = np.array(polygon, dtype=np.int32)
    # Mask: black outside polygon
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    # Crop to bounding box
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return masked[y1:y2, x1:x2]


def main():
    db = SessionLocal()
    try:
        cameras = db.query(Camera).filter(Camera.is_active == True).all()
        if not cameras:
            print("No active cameras.")
            return

        for cam in cameras:
            slots = [s for s in cam.slots if s.polygon_coords]
            if not slots:
                print(f"[{cam.label}] No slots, skipping.")
                continue

            print(f"[{cam.label}] Capturing from {cam.source} ...")
            cap = cv2.VideoCapture(cam.source)
            if not cap.isOpened():
                print(f"[{cam.label}] Failed to open.")
                continue
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                print(f"[{cam.label}] Failed to capture.")
                continue

            h, w = frame.shape[:2]
            print(f"[{cam.label}] Frame: {w}x{h}, Slots: {len(slots)}")

            for slot in slots:
                polygon = slot.polygon_coords
                if isinstance(polygon, str):
                    polygon = json.loads(polygon)
                if not polygon:
                    continue

                # Draw polygon on crop for reference
                cropped = crop_polygon(frame, polygon)
                ch, cw = cropped.shape[:2]

                safe = slot.label.replace(" ", "_").replace("/", "_")
                path = os.path.join(OUT_DIR, f"{safe}.jpg")
                cv2.imwrite(path, cropped)
                print(f"  {slot.label}: {cw}x{ch} -> {path}")

    finally:
        db.close()

    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
