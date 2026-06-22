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
    """Crop frame to polygon shape with transparent background."""
    pts = np.array(polygon, dtype=np.int32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    # Create alpha mask from polygon
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    # Crop to bbox
    crop = frame[y1:y2, x1:x2]
    alpha = mask[y1:y2, x1:x2]
    # Merge BGR + Alpha → BGRA
    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


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

                pts = np.array(polygon, dtype=np.int32)
                px1, py1 = pts.min(axis=0)
                px2, py2 = pts.max(axis=0)
                print(f"  {slot.label}: polygon bbox=({px1},{py1})->({px2},{py2}) frame={w}x{h}")

                cropped = crop_polygon(frame, polygon)
                ch, cw = cropped.shape[:2]

                safe = slot.label.replace(" ", "_").replace("/", "_")
                path = os.path.join(OUT_DIR, f"{safe}.png")
                cv2.imwrite(path, cropped)
                print(f"    cropped: {cw}x{ch} ({cw*ch*100//(w*h)}% of frame) -> {path}")

    finally:
        db.close()

    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
