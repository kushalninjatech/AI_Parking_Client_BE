"""Test script: Crop each slot ROI from frame, run YOLO on the crop.

For each slot polygon:
  1. Crop the bounding box region (with padding) from the full frame
  2. Run YOLO on just that crop
  3. Compare: did full-frame YOLO detect it? Did slot-crop YOLO detect it?

Usage (on RPi):
    python3 test_yolo_slot_crop.py
    python3 test_yolo_slot_crop.py --padding 50     # extra pixels around polygon bbox

Output:
    data/debug/slot_crop_{label}.jpg        — each slot crop with detections
    data/debug/test_slot_crop_summary.jpg   — full frame with results
"""

import argparse
import json
import os
import cv2
import numpy as np
from app.core.config import settings
from app.detection.yolo_detector import YOLODetector
from app.db.session import SessionLocal
from app.models.camera import Camera
from app.models.parking_slot import ParkingSlot
from app.camera.factory import create_camera

COCO_VEHICLE_MAP = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
OVERLAP_THRESHOLD = settings.SLOT_OVERLAP_THRESHOLD


def compute_bbox_overlap(det_bbox, slot_polygon_np):
    """Fraction of detection bbox area that overlaps with slot polygon."""
    bx1, by1, bx2, by2 = det_bbox
    bbox_poly = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32)
    slot_hull = cv2.convexHull(slot_polygon_np.astype(np.float32))
    inter_area, _ = cv2.intersectConvexConvex(bbox_poly, slot_hull)
    bbox_area = (bx2 - bx1) * (by2 - by1)
    if bbox_area <= 0:
        return 0.0
    return inter_area / bbox_area


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--padding", type=int, default=30, help="Extra pixels around slot bbox for crop")
    args = parser.parse_args()

    print(f"YOLO model: {settings.YOLO_MODEL_PATH}")
    print(f"YOLO input size: {settings.YOLO_INPUT_SIZE}")
    print(f"YOLO confidence: {settings.YOLO_CONFIDENCE}")
    print(f"Crop padding: {args.padding}px")

    # Load YOLO
    yolo = YOLODetector()
    if not yolo.load():
        print("ERROR: Failed to load YOLO model")
        exit(1)

    # Get camera + slots
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

    img_h, img_w = frame.shape[:2]
    print(f"Frame size: {img_w}x{img_h}")

    # === Full-frame YOLO (baseline) ===
    print(f"\n{'='*60}")
    print("FULL-FRAME YOLO (baseline)")
    print(f"{'='*60}")
    full_dets = yolo.detect(frame, camera_label=f"{cam.label}_full")
    print(f"Full-frame: {len(full_dets)} vehicles detected\n")

    # === Per-slot crop YOLO ===
    print(f"{'='*60}")
    print("PER-SLOT CROP YOLO")
    print(f"{'='*60}\n")

    os.makedirs("data/debug", exist_ok=True)
    summary = frame.copy()
    pad = args.padding

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

        # Crop region with padding
        cx1 = max(0, px_min - pad)
        cy1 = max(0, py_min - pad)
        cx2 = min(img_w, px_max + pad)
        cy2 = min(img_h, py_max + pad)
        crop = frame[cy1:cy2, cx1:cx2]
        crop_h, crop_w = crop.shape[:2]

        print(f"Slot {slot.label}:")
        print(f"  Polygon: ({px_min},{py_min})-({px_max},{py_max})")
        print(f"  Crop:    ({cx1},{cy1})-({cx2},{cy2}) = {crop_w}x{crop_h}")

        # Full-frame matching for this slot (bbox overlap)
        full_matches = []
        for det in full_dets:
            overlap = compute_bbox_overlap(det["bbox"], pts)
            if overlap >= OVERLAP_THRESHOLD:
                full_matches.append((det, overlap))

        if full_matches:
            for det, overlap in full_matches:
                vname = COCO_VEHICLE_MAP.get(det["class_id"], "?")
                print(f"  [FULL]  MATCH: {vname} conf={det['confidence']:.2f} overlap={overlap:.1%}")
        else:
            print(f"  [FULL]  NO MATCH")

        # Run YOLO on slot crop
        crop_dets = yolo.detect(crop, camera_label=f"slot_{slot.label}")

        # Remap crop detections to full-frame coords
        crop_dets_remapped = []
        for det in crop_dets:
            x1, y1, x2, y2 = det["bbox"]
            cx_det, cy_det = det["centroid"]
            remapped = {
                **det,
                "bbox": [x1 + cx1, y1 + cy1, x2 + cx1, y2 + cy1],
                "centroid": [cx_det + cx1, cy_det + cy1],
            }
            crop_dets_remapped.append(remapped)

        # Match remapped crop detections to this slot
        crop_matches = []
        for det in crop_dets_remapped:
            overlap = compute_bbox_overlap(det["bbox"], pts)
            if overlap >= OVERLAP_THRESHOLD:
                crop_matches.append((det, overlap))

        if crop_matches:
            for det, overlap in crop_matches:
                vname = COCO_VEHICLE_MAP.get(det["class_id"], "?")
                x1, y1, x2, y2 = det["bbox"]
                print(f"  [CROP]  MATCH: {vname} conf={det['confidence']:.2f} "
                      f"bbox=({x1},{y1})-({x2},{y2}) overlap={overlap:.1%}")
        else:
            print(f"  [CROP]  NO MATCH ({len(crop_dets)} raw detections in crop)")

        # Verdict
        full_found = len(full_matches) > 0
        crop_found = len(crop_matches) > 0
        if crop_found and not full_found:
            verdict = "CROP RESCUED"
        elif full_found and crop_found:
            verdict = "BOTH FOUND"
        elif full_found and not crop_found:
            verdict = "FULL ONLY"
        else:
            verdict = "NEITHER"
        print(f"  >>> {verdict}")
        print()

        # Draw on summary image
        if crop_found:
            color = (0, 255, 0)   # green = crop found it
        elif full_found:
            color = (255, 255, 0) # cyan = full found it
        else:
            color = (0, 0, 255)   # red = neither
        cv2.polylines(summary, [pts], True, color, 3)
        cx_p, cy_p = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        cv2.putText(summary, f"{slot.label}: {verdict}", (cx_p - 80, cy_p),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Save individual slot crop with detections drawn
        crop_debug = crop.copy()
        # Draw polygon on crop (shifted to crop coords)
        pts_crop = pts.copy()
        pts_crop[:, 0] -= cx1
        pts_crop[:, 1] -= cy1
        cv2.polylines(crop_debug, [pts_crop], True, (0, 255, 0), 2)

        for det in crop_dets:
            x1, y1, x2, y2 = det["bbox"]
            cls = det["class_id"]
            conf = det["confidence"]
            bcolor = (0, 0, 255) if cls == 2 else (255, 100, 0)
            cv2.rectangle(crop_debug, (x1, y1), (x2, y2), bcolor, 2)
            label = f"{COCO_VEHICLE_MAP.get(cls, '?')} {conf:.2f}"
            cv2.putText(crop_debug, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bcolor, 2)

        cv2.putText(crop_debug, f"{slot.label}: {verdict}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        crop_path = f"data/debug/slot_crop_{slot.label}.jpg"
        cv2.imwrite(crop_path, crop_debug)

    # Draw full-frame detections on summary
    for det in full_dets:
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class_id"]
        bcolor = (0, 0, 255) if cls == 2 else (255, 100, 0)
        cv2.rectangle(summary, (x1, y1), (x2, y2), bcolor, 1)

    summary_path = "data/debug/test_slot_crop_summary.jpg"
    cv2.imwrite(summary_path, summary)

    print(f"{'='*60}")
    print(f"Output saved:")
    print(f"  {summary_path} — full frame with verdicts")
    print(f"  data/debug/slot_crop_*.jpg — individual slot crops")
    db.close()


if __name__ == "__main__":
    main()
