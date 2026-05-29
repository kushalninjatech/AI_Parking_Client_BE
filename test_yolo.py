"""Test script: Capture frame, run YOLO, draw all detections + slot polygons, save output image.

Usage (on RPi):
    python3 test_yolo.py

Output:
    data/debug/test_output.jpg  — annotated image with all bboxes + polygons
"""

import json
import cv2
import numpy as np
from app.core.config import settings
from app.detection.yolo_detector import YOLODetector
from app.db.session import SessionLocal
from app.models.camera import Camera
from app.models.parking_slot import ParkingSlot
from app.camera.factory import create_camera

print(f"YOLO model: {settings.YOLO_MODEL_PATH}")
print(f"YOLO input size: {settings.YOLO_INPUT_SIZE}")
print(f"YOLO confidence: {settings.YOLO_CONFIDENCE}")

# Load YOLO
yolo = YOLODetector()
if not yolo.load():
    print("ERROR: Failed to load YOLO model")
    exit(1)

# Get first active camera + its slots
db = SessionLocal()
cam = db.query(Camera).filter(Camera.is_active == True).first()
if not cam:
    print("ERROR: No active camera found")
    exit(1)

slots = db.query(ParkingSlot).filter(ParkingSlot.camera_id == cam.id).all()
print(f"\nCamera: {cam.label} ({cam.source})")
print(f"Slots: {len(slots)}")

# Capture frame
print("\nCapturing frame...")
camera = create_camera(cam.source, cam.camera_type)
if not camera.open():
    print("ERROR: Failed to open camera")
    exit(1)

frame = None
for _ in range(5):
    ok, frame = camera.read()
    if ok and frame is not None:
        break
camera.release()

if frame is None:
    print("ERROR: Failed to capture frame")
    exit(1)

h, w = frame.shape[:2]
print(f"Frame size: {w}x{h}")

# Run YOLO
print("\nRunning YOLO detection...")
detections = yolo.detect(frame, camera_label=cam.label)
print(f"Detected {len(detections)} vehicles\n")

# Draw detections on frame
output = frame.copy()

# Draw all vehicle bboxes in RED with centroid
for i, det in enumerate(detections):
    x1, y1, x2, y2 = det["bbox"]
    cx, cy = det["centroid"]
    cls = det["class_id"]
    conf = det["confidence"]
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.circle(output, (cx, cy), 8, (0, 0, 255), -1)
    label = f"cls{cls} {conf:.2f} ({cx},{cy})"
    cv2.putText(output, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    print(f"  Vehicle {i+1}: class={cls} conf={conf:.2f} bbox=({x1},{y1})-({x2},{y2}) centroid=({cx},{cy})")

# Draw slot polygons in GREEN
print(f"\nSlot polygons:")
for slot in slots:
    if not slot.polygon_coords:
        print(f"  {slot.label}: NO POLYGON")
        continue
    coords = slot.polygon_coords
    if isinstance(coords, str):
        coords = json.loads(coords)
    pts = np.array(coords, dtype=np.int32)
    px_min, py_min = pts.min(axis=0)
    px_max, py_max = pts.max(axis=0)
    print(f"  {slot.label}: polygon=({px_min},{py_min})-({px_max},{py_max})")

    cv2.polylines(output, [pts], True, (0, 255, 0), 3)
    cx_p = int(pts[:, 0].mean())
    cy_p = int(pts[:, 1].mean())
    cv2.putText(output, slot.label, (cx_p - 30, cy_p), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Check which vehicles match this polygon
    matched = 0
    for det in detections:
        cx, cy = det["centroid"]
        if cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0:
            matched += 1
            print(f"    -> CENTROID MATCH: class={det['class_id']} conf={det['confidence']:.2f} centroid=({cx},{cy})")
    if matched == 0:
        print(f"    -> NO CENTROID MATCHES")

# Save
import os
os.makedirs("data/debug", exist_ok=True)
path = "data/debug/test_output.jpg"
cv2.imwrite(path, output)
print(f"\nOutput saved: {path}")
print("Transfer to your machine to view: scp rpi:~/.../{path} .")
db.close()
