"""Test script: Full-frame + bottom-crop YOLO detection with bbox-overlap matching.

Runs YOLO twice:
  1. Full frame — catches mid/far vehicles
  2. Bottom 50% crop — catches near-camera vehicles that full-frame misses

Deduplicates detections between the two passes using IoU.

Usage (on RPi):
    python3 test_yolo_crop.py
    python3 test_yolo_crop.py --crop-ratio 0.5       # bottom 50% (default)
    python3 test_yolo_crop.py --crop-ratio 0.6       # bottom 60%
    python3 test_yolo_crop.py --crop-overlap 0.3     # crop starts 30% from center

Output:
    data/debug/test_crop_full.jpg      — full-frame detections only
    data/debug/test_crop_bottom.jpg    — bottom-crop detections only
    data/debug/test_crop_merged.jpg    — merged (deduplicated) + slot matching
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


def compute_iou(box1, box2):
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


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


def remap_crop_detections(detections, crop_y_offset):
    """Shift crop detections back to full-frame coordinates."""
    remapped = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = det["centroid"]
        remapped.append({
            **det,
            "bbox": [x1, y1 + crop_y_offset, x2, y2 + crop_y_offset],
            "centroid": [cx, cy + crop_y_offset],
            "source": "crop",
        })
    return remapped


def merge_detections(full_dets, crop_dets, iou_threshold=0.4):
    """Merge full-frame and crop detections, removing duplicates."""
    merged = [{**d, "source": "full"} for d in full_dets]

    for crop_det in crop_dets:
        is_duplicate = False
        for full_det in merged:
            iou = compute_iou(crop_det["bbox"], full_det["bbox"])
            if iou > iou_threshold:
                # Keep the one with higher confidence
                if crop_det["confidence"] > full_det["confidence"]:
                    full_det.update(crop_det)
                is_duplicate = True
                break
        if not is_duplicate:
            merged.append(crop_det)

    return merged


def draw_detections(frame, detections, title=""):
    """Draw bboxes + centroids on frame."""
    output = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = det["centroid"]
        cls = det["class_id"]
        conf = det["confidence"]
        source = det.get("source", "")

        if source == "crop":
            color = (0, 255, 0)  # green = from crop
        elif source == "full":
            color = (0, 0, 255) if cls == 2 else (255, 100, 0)  # red=car, blue=2w
        else:
            color = (0, 0, 255) if cls == 2 else (255, 100, 0)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.circle(output, (cx, cy), 5, color, -1)
        label = f"{COCO_VEHICLE_MAP.get(cls, '?')} {conf:.2f}"
        if source:
            label += f" [{source}]"
        cv2.putText(output, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if title:
        cv2.putText(output, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-ratio", type=float, default=0.5, help="Bottom portion to crop (0.5 = bottom 50%%)")
    parser.add_argument("--crop-overlap", type=float, default=0.0, help="Extra overlap from center (0.0-0.3)")
    args = parser.parse_args()

    print(f"YOLO model: {settings.YOLO_MODEL_PATH}")
    print(f"YOLO input size: {settings.YOLO_INPUT_SIZE}")
    print(f"YOLO confidence: {settings.YOLO_CONFIDENCE}")
    print(f"Crop ratio: {args.crop_ratio}")
    print(f"Overlap threshold: {OVERLAP_THRESHOLD}")

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

    # === Pass 1: Full frame ===
    print(f"\n{'='*60}")
    print("PASS 1: Full frame")
    print(f"{'='*60}")
    full_dets = yolo.detect(frame, camera_label=f"{cam.label}_full")
    print(f"Full-frame detections: {len(full_dets)}")
    for i, det in enumerate(full_dets):
        x1, y1, x2, y2 = det["bbox"]
        print(f"  {i+1}. {COCO_VEHICLE_MAP.get(det['class_id'], '?'):12s} "
              f"conf={det['confidence']:.2f} bbox=({x1},{y1})-({x2},{y2})")

    # === Pass 2: Bottom crop ===
    crop_start_y = int(img_h * (1.0 - args.crop_ratio - args.crop_overlap))
    crop_start_y = max(0, crop_start_y)
    crop_frame = frame[crop_start_y:, :]
    crop_h, crop_w = crop_frame.shape[:2]

    print(f"\n{'='*60}")
    print(f"PASS 2: Bottom crop (y={crop_start_y} to {img_h}, size={crop_w}x{crop_h})")
    print(f"{'='*60}")
    crop_dets_raw = yolo.detect(crop_frame, camera_label=f"{cam.label}_crop")
    crop_dets = remap_crop_detections(crop_dets_raw, crop_start_y)
    print(f"Crop detections: {len(crop_dets)}")
    for i, det in enumerate(crop_dets):
        x1, y1, x2, y2 = det["bbox"]
        print(f"  {i+1}. {COCO_VEHICLE_MAP.get(det['class_id'], '?'):12s} "
              f"conf={det['confidence']:.2f} bbox=({x1},{y1})-({x2},{y2}) [remapped]")

    # === Merge ===
    merged = merge_detections(full_dets, crop_dets)
    new_from_crop = sum(1 for d in merged if d.get("source") == "crop")

    print(f"\n{'='*60}")
    print(f"MERGED: {len(merged)} total ({len(full_dets)} full + {new_from_crop} new from crop)")
    print(f"{'='*60}")
    for i, det in enumerate(merged):
        x1, y1, x2, y2 = det["bbox"]
        bw, bh = x2 - x1, y2 - y1
        frame_pct = (bw * bh) / (img_w * img_h) * 100
        src = det.get("source", "?")
        print(f"  {i+1}. {COCO_VEHICLE_MAP.get(det['class_id'], '?'):12s} "
              f"conf={det['confidence']:.2f} bbox=({x1},{y1})-({x2},{y2}) "
              f"size={bw}x{bh} frame%={frame_pct:.1f}% [{src}]")

    # === Slot matching with bbox overlap ===
    print(f"\n{'='*60}")
    print(f"SLOT MATCHING (bbox overlap >= {OVERLAP_THRESHOLD:.0%})")
    print(f"{'='*60}\n")

    output_merged = frame.copy()

    for slot in slots:
        if not slot.polygon_coords:
            continue
        coords = slot.polygon_coords
        if isinstance(coords, str):
            coords = json.loads(coords)
        if not coords:
            continue

        pts = np.array(coords, dtype=np.int32)
        print(f"Slot {slot.label}:")

        matched = []
        for det in merged:
            overlap = compute_bbox_overlap(det["bbox"], pts)
            if overlap >= OVERLAP_THRESHOLD:
                matched.append((det, overlap))

        if matched:
            for det, overlap in matched:
                vname = COCO_VEHICLE_MAP.get(det["class_id"], "?")
                x1, y1, x2, y2 = det["bbox"]
                src = det.get("source", "?")
                print(f"  MATCH: {vname} conf={det['confidence']:.2f} "
                      f"bbox=({x1},{y1})-({x2},{y2}) overlap={overlap:.1%} [{src}]")
            color = (0, 255, 0)
            state = "VEHICLE"
        else:
            print(f"  NO MATCH")
            color = (0, 0, 255)
            state = "EMPTY"
        print()

        cv2.polylines(output_merged, [pts], True, color, 3)
        cx_p, cy_p = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        cv2.putText(output_merged, f"{slot.label}: {state}", (cx_p - 60, cy_p),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw merged detections
    for det in merged:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = det["centroid"]
        cls = det["class_id"]
        conf = det["confidence"]
        src = det.get("source", "")
        color = (0, 255, 0) if src == "crop" else ((0, 0, 255) if cls == 2 else (255, 100, 0))
        cv2.rectangle(output_merged, (x1, y1), (x2, y2), color, 2)
        cv2.circle(output_merged, (cx, cy), 5, color, -1)
        label = f"{COCO_VEHICLE_MAP.get(cls, '?')} {conf:.2f} [{src}]"
        cv2.putText(output_merged, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw crop boundary line
    cv2.line(output_merged, (0, crop_start_y), (img_w, crop_start_y), (255, 255, 0), 2)
    cv2.putText(output_merged, f"crop line (y={crop_start_y})", (10, crop_start_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Save all outputs
    os.makedirs("data/debug", exist_ok=True)

    out_full = draw_detections(frame, [{**d, "source": "full"} for d in full_dets],
                               f"FULL FRAME: {len(full_dets)} detections")
    cv2.imwrite("data/debug/test_crop_full.jpg", out_full)

    out_crop = draw_detections(frame, crop_dets,
                               f"BOTTOM CROP: {len(crop_dets)} detections (remapped)")
    cv2.line(out_crop, (0, crop_start_y), (img_w, crop_start_y), (255, 255, 0), 2)
    cv2.imwrite("data/debug/test_crop_bottom.jpg", out_crop)

    cv2.putText(output_merged,
                f"MERGED: {len(full_dets)} full + {new_from_crop} new from crop = {len(merged)} total",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite("data/debug/test_crop_merged.jpg", output_merged)

    print(f"Output saved:")
    print(f"  data/debug/test_crop_full.jpg    — full-frame only")
    print(f"  data/debug/test_crop_bottom.jpg  — crop detections on full frame")
    print(f"  data/debug/test_crop_merged.jpg  — merged + slot matching")
    db.close()


if __name__ == "__main__":
    main()
