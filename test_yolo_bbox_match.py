"""Test script: YOLO detection + bbox-overlap slot matching (30% threshold).

Compares centroid-in-polygon vs bbox-overlap matching side by side.

Usage (on RPi):
    python3 test_yolo_bbox_match.py

Output:
    data/debug/test_bbox_match.jpg — annotated image
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

OVERLAP_THRESHOLD = settings.SLOT_OVERLAP_THRESHOLD  # 0.3 from config
COCO_VEHICLE_MAP = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

print(f"YOLO model: {settings.YOLO_MODEL_PATH}")
print(f"YOLO input size: {settings.YOLO_INPUT_SIZE}")
print(f"YOLO confidence: {settings.YOLO_CONFIDENCE}")
print(f"Overlap threshold: {OVERLAP_THRESHOLD}")

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

# Print all detections
for i, det in enumerate(detections):
    x1, y1, x2, y2 = det["bbox"]
    bw, bh = x2 - x1, y2 - y1
    frame_pct = (bw * bh) / (w * h) * 100
    print(f"  Vehicle {i+1}: {COCO_VEHICLE_MAP.get(det['class_id'], '?'):12s} "
          f"conf={det['confidence']:.2f} bbox=({x1},{y1})-({x2},{y2}) "
          f"size={bw}x{bh} frame%={frame_pct:.1f}% centroid=({det['centroid'][0]},{det['centroid'][1]})")


def compute_bbox_overlap(det_bbox, slot_polygon_np):
    """Compute what fraction of the detection bbox area overlaps with the slot polygon."""
    bx1, by1, bx2, by2 = det_bbox
    bbox_poly = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32)
    slot_poly = slot_polygon_np.astype(np.float32)

    # intersectConvexConvex needs convex polygons — bbox is always convex
    # For slot polygon, use convex hull as approximation
    slot_hull = cv2.convexHull(slot_poly)
    inter_area, _ = cv2.intersectConvexConvex(bbox_poly, slot_hull)

    bbox_area = (bx2 - bx1) * (by2 - by1)
    if bbox_area <= 0:
        return 0.0
    return inter_area / bbox_area


# Compare matching methods per slot
output = frame.copy()
print(f"\n{'='*80}")
print(f"SLOT MATCHING COMPARISON: centroid-in-polygon vs bbox-overlap ({OVERLAP_THRESHOLD:.0%})")
print(f"{'='*80}\n")

for slot in slots:
    if not slot.polygon_coords:
        continue
    coords = slot.polygon_coords
    if isinstance(coords, str):
        coords = json.loads(coords)
    if not coords:
        continue

    pts = np.array(coords, dtype=np.int32)
    px_min, py_min = pts.min(axis=0)
    px_max, py_max = pts.max(axis=0)

    print(f"Slot {slot.label}: polygon=({px_min},{py_min})-({px_max},{py_max})")

    centroid_matches = []
    bbox_matches = []

    for det in detections:
        cx, cy = det["centroid"]
        vname = COCO_VEHICLE_MAP.get(det["class_id"], "?")

        # Method 1: centroid-in-polygon (current)
        if cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0:
            centroid_matches.append(det)

        # Method 2: bbox overlap
        overlap = compute_bbox_overlap(det["bbox"], pts)
        if overlap >= OVERLAP_THRESHOLD:
            bbox_matches.append((det, overlap))

    # Report centroid matches
    if centroid_matches:
        for det in centroid_matches:
            vname = COCO_VEHICLE_MAP.get(det["class_id"], "?")
            print(f"  [CENTROID]  MATCH: {vname} conf={det['confidence']:.2f} centroid=({det['centroid'][0]},{det['centroid'][1]})")
    else:
        print(f"  [CENTROID]  NO MATCH")

    # Report bbox matches
    if bbox_matches:
        for det, overlap in bbox_matches:
            vname = COCO_VEHICLE_MAP.get(det["class_id"], "?")
            x1, y1, x2, y2 = det["bbox"]
            print(f"  [BBOX {OVERLAP_THRESHOLD:.0%}]  MATCH: {vname} conf={det['confidence']:.2f} "
                  f"bbox=({x1},{y1})-({x2},{y2}) overlap={overlap:.1%}")
    else:
        print(f"  [BBOX {OVERLAP_THRESHOLD:.0%}]  NO MATCH")

    # Highlight differences
    centroid_set = {id(d) for d in centroid_matches}
    bbox_set = {id(d) for d, _ in bbox_matches}
    only_bbox = [(d, o) for d, o in bbox_matches if id(d) not in centroid_set]
    only_centroid = [d for d in centroid_matches if id(d) not in bbox_set]

    if only_bbox:
        print(f"  >>> BBOX FOUND {len(only_bbox)} EXTRA (missed by centroid):")
        for det, overlap in only_bbox:
            vname = COCO_VEHICLE_MAP.get(det["class_id"], "?")
            print(f"      + {vname} conf={det['confidence']:.2f} overlap={overlap:.1%}")
    if only_centroid:
        print(f"  >>> CENTROID FOUND {len(only_centroid)} EXTRA (missed by bbox):")
        for det in only_centroid:
            vname = COCO_VEHICLE_MAP.get(det["class_id"], "?")
            print(f"      + {vname} conf={det['confidence']:.2f}")
    print()

    # Draw slot polygon — green if bbox matched, yellow if only centroid, red if no match
    if bbox_matches:
        color = (0, 255, 0)
        state = "VEHICLE"
    elif centroid_matches:
        color = (0, 255, 255)
        state = "VEHICLE(centroid-only)"
    else:
        color = (0, 0, 255)
        state = "EMPTY"
    cv2.polylines(output, [pts], True, color, 3)
    cx_p, cy_p = int(pts[:, 0].mean()), int(pts[:, 1].mean())
    cv2.putText(output, f"{slot.label}: {state}", (cx_p - 60, cy_p), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Draw all vehicle bboxes
for det in detections:
    x1, y1, x2, y2 = det["bbox"]
    cx, cy = det["centroid"]
    cls = det["class_id"]
    conf = det["confidence"]
    bcolor = (0, 0, 255) if cls == 2 else (255, 100, 0)
    cv2.rectangle(output, (x1, y1), (x2, y2), bcolor, 2)
    cv2.circle(output, (cx, cy), 5, bcolor, -1)
    label = f"{COCO_VEHICLE_MAP.get(cls, '?')} {conf:.2f}"
    cv2.putText(output, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bcolor, 2)

# Save
import os
os.makedirs("data/debug", exist_ok=True)
path = "data/debug/test_bbox_match.jpg"
cv2.imwrite(path, output)
print(f"\nOutput saved: {path}")
db.close()
