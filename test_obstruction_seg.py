"""Test obstruction detection using YOLOv11x-seg (instance segmentation) + Depth.

Pipeline:
  1. Run YOLO-seg → pixel-level masks for cars, 2W, persons
  2. Run Depth → full depth map
  3. Build ground plane reference (per-row linear fit)
  4. Find "elevated" pixels (something above ground)
  5. Subtract YOLO segmentation masks (known objects)
  6. Remaining elevated area = obstruction

Usage:
    python3 test_obstruction_seg.py --image raw_frame.jpg --no-slots
    python3 test_obstruction_seg.py --image raw_frame.jpg --polygon "x1,y1;x2,y2;x3,y3;x4,y4"
    python3 test_obstruction_seg.py --image raw_frame.jpg --no-slots --depth-thresh 0.15

Install: pip install ultralytics onnxruntime opencv-python numpy
"""

import argparse
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

# COCO classes
CLASS_MAP = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
KNOWN_VEHICLE_IDS = {2, 3, 5, 7}
PERSON_ID = 0

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COLORS = {
    "person": (0, 255, 0),
    "car": (0, 0, 255),
    "motorcycle": (255, 100, 0),
    "bus": (0, 180, 0),
    "truck": (180, 0, 180),
}


def run_yolo_seg(model, frame, conf=0.25):
    """Run YOLO-seg, return detections with pixel masks."""
    h, w = frame.shape[:2]
    t0 = time.monotonic()
    results = model(frame, imgsz=640, conf=conf, verbose=False)[0]
    elapsed = time.monotonic() - t0

    detections = []
    masks_combined = {
        "vehicle": np.zeros((h, w), dtype=np.uint8),  # cars + 2W + bus + truck
        "person": np.zeros((h, w), dtype=np.uint8),
        "all": np.zeros((h, w), dtype=np.uint8),       # everything known
    }

    if results.masks is None:
        return detections, masks_combined, elapsed

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        if cls_id not in CLASS_MAP:
            continue

        c = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        # Get pixel mask for this detection, resize to frame size
        mask_tensor = results.masks.data[i].cpu().numpy()
        mask = cv2.resize(mask_tensor, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask > 0.5).astype(np.uint8) * 255

        detections.append({
            "class_id": cls_id,
            "class_name": CLASS_MAP[cls_id],
            "confidence": c,
            "bbox": [x1, y1, x2, y2],
            "mask": mask_binary,
        })

        # Accumulate into combined masks
        masks_combined["all"] = cv2.bitwise_or(masks_combined["all"], mask_binary)
        if cls_id in KNOWN_VEHICLE_IDS:
            masks_combined["vehicle"] = cv2.bitwise_or(masks_combined["vehicle"], mask_binary)
        elif cls_id == PERSON_ID:
            masks_combined["person"] = cv2.bitwise_or(masks_combined["person"], mask_binary)

    return detections, masks_combined, elapsed


def run_depth(frame, session, input_size=384):
    """Run depth estimation, return normalized depth map."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size))
    normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    blob = normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    input_name = session.get_inputs()[0].name
    t0 = time.monotonic()
    output = session.run(None, {input_name: blob})[0]
    elapsed = time.monotonic() - t0

    depth = output.squeeze()
    depth = cv2.resize(depth, (w, h))
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
    return depth_norm, elapsed


def build_ground_plane(depth_norm, poly_mask, yolo_all_mask, h, w, py, ph):
    """Fit linear ground plane from depth samples, excluding YOLO objects.
    Uses p25 (25th percentile) to get actual ground surface, ignoring objects above.
    """
    # Expand YOLO mask slightly for ground sampling exclusion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    exclude_mask = cv2.dilate(yolo_all_mask, kernel)

    num_strips = 20
    strip_rows = []
    strip_depths = []

    for i in range(num_strips):
        row_y = py + int(ph * i / num_strips)
        row_y_end = min(row_y + max(ph // num_strips, 3), py + ph)
        strip = poly_mask[row_y:row_y_end, :].copy()
        strip[exclude_mask[row_y:row_y_end, :] > 0] = 0
        pixels = depth_norm[row_y:row_y_end, :][strip > 0]
        if len(pixels) > 5:
            strip_rows.append((row_y + row_y_end) // 2)
            strip_depths.append(float(np.percentile(pixels, 25)))

    if len(strip_rows) >= 3:
        coeffs = np.polyfit(strip_rows, strip_depths, 1)
        ground_plane = np.polyval(coeffs, np.arange(h)).astype(np.float32)
    elif len(strip_rows) >= 1:
        ground_plane = np.full(h, np.mean(strip_depths), dtype=np.float32)
    else:
        # Fallback: linear from top to bottom of frame
        top_med = float(np.median(depth_norm[:int(h * 0.2), :]))
        bot_med = float(np.median(depth_norm[int(h * 0.8):, :]))
        ground_plane = np.linspace(top_med, bot_med, h).astype(np.float32)

    return ground_plane


def detect_obstruction(depth_norm, detections, masks_combined, polygon, depth_thresh, obstruct_thresh):
    """Detect obstruction using segmentation masks + depth subtraction."""
    h, w = depth_norm.shape[:2]
    poly_np = np.array(polygon, dtype=np.int32)

    # Polygon mask
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [poly_np], 255)
    poly_area = cv2.countNonZero(poly_mask)
    if poly_area == 0:
        return {"obstructed": False, "reason": "empty polygon"}

    px, py, pw, ph = cv2.boundingRect(poly_np)

    # Build ground plane (excluding ALL known objects)
    ground_plane = build_ground_plane(depth_norm, poly_mask, masks_combined["all"], h, w, py, ph)
    ground_ref = float(np.mean(ground_plane[py:py + ph]))

    # 2D ground reference
    ground_2d = np.repeat(ground_plane[:, np.newaxis], w, axis=1)

    # Elevated pixels = deviate from ground plane
    depth_float = depth_norm.astype(np.float32)
    deviation = np.abs(depth_float - ground_2d) / 255.0
    elevated_mask = (deviation > depth_thresh) & (poly_mask > 0)
    elevated_area = cv2.countNonZero(elevated_mask.astype(np.uint8))

    # Dilate vehicle masks slightly to cover wheels, shadows, and edges
    # that segmentation misses but depth still sees as elevated
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    vehicle_dilated = cv2.dilate(masks_combined["vehicle"], dilate_kernel)
    person_dilated = cv2.dilate(masks_combined["person"], dilate_kernel)

    # Count vehicles/persons inside polygon (using dilated segmentation masks)
    vehicle_in_poly = cv2.bitwise_and(vehicle_dilated, poly_mask)
    person_in_poly = cv2.bitwise_and(person_dilated, poly_mask)
    all_known_in_poly = cv2.bitwise_and(masks_combined["all"], poly_mask)

    vehicle_area = cv2.countNonZero(vehicle_in_poly)
    person_area = cv2.countNonZero(person_in_poly)
    known_area = cv2.countNonZero(all_known_in_poly)

    # Count individual vehicles by checking centroid in polygon
    cars = 0
    two_wheelers = 0
    persons = 0
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if cv2.pointPolygonTest(poly_np.astype(np.float32), (float(cx), float(cy)), False) >= 0:
            if det["class_id"] == 2:
                cars += 1
            elif det["class_id"] == 3:
                two_wheelers += 1
            elif det["class_id"] == PERSON_ID:
                persons += 1

    # Obstruction = elevated AND NOT covered by any YOLO segmentation mask (dilated)
    # Persons are excluded from "explaining" area — they're temporary and might be hiding obstructions
    obstruction_mask = elevated_mask & (vehicle_in_poly == 0) & (poly_mask > 0)

    # Also exclude person pixels from obstruction (person themselves aren't obstructions)
    obstruction_mask = obstruction_mask & (person_in_poly == 0)

    obstruction_area = cv2.countNonZero(obstruction_mask.astype(np.uint8))
    obstruction_ratio = obstruction_area / poly_area if poly_area > 0 else 0
    elevated_ratio = elevated_area / poly_area if poly_area > 0 else 0
    vehicle_coverage = vehicle_area / poly_area if poly_area > 0 else 0
    known_coverage = known_area / poly_area if poly_area > 0 else 0

    is_obstructed = obstruction_ratio > obstruct_thresh

    return {
        "obstructed": is_obstructed,
        "obstruction_ratio": obstruction_ratio,
        "elevated_ratio": elevated_ratio,
        "vehicle_coverage": vehicle_coverage,
        "known_coverage": known_coverage,
        "obstruction_area": obstruction_area,
        "elevated_area": elevated_area,
        "vehicle_area": vehicle_area,
        "poly_area": poly_area,
        "ground_ref": ground_ref,
        "cars": cars,
        "two_wheelers": two_wheelers,
        "persons": persons,
        "obstruction_mask": obstruction_mask,
        "vehicle_mask": vehicle_in_poly,
        "person_mask": person_in_poly,
    }


def draw_results(frame, depth_norm, detections, masks_combined, analysis, polygon):
    """Draw 3-panel visualization."""
    h, w = frame.shape[:2]
    poly_np = np.array(polygon, dtype=np.int32)

    # --- Panel 1: Original + segmentation overlay + annotations ---
    output = frame.copy()

    # Draw segmentation masks with transparency
    seg_overlay = output.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if cv2.pointPolygonTest(poly_np.astype(np.float32), (float(cx), float(cy)), False) >= 0:
            color = COLORS.get(det["class_name"], (255, 255, 255))
            seg_overlay[det["mask"] > 0] = color
    output = cv2.addWeighted(output, 0.6, seg_overlay, 0.4, 0)

    # Draw polygon border
    poly_color = (0, 0, 255) if analysis["obstructed"] else (0, 255, 0)
    cv2.polylines(output, [poly_np], True, poly_color, 3)

    # Overlay obstruction in bright red
    if analysis["obstructed"]:
        obstruct_overlay = output.copy()
        obstruct_overlay[analysis["obstruction_mask"]] = (0, 0, 255)
        output = cv2.addWeighted(output, 0.5, obstruct_overlay, 0.5, 0)

    # Text
    status = "OBSTRUCTED" if analysis["obstructed"] else "CLEAR"
    status_color = (0, 0, 255) if analysis["obstructed"] else (0, 200, 0)
    cv2.putText(output, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    cv2.putText(output, f"Cars: {analysis['cars']}  2W: {analysis['two_wheelers']}  Persons: {analysis['persons']}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(output, f"Obstruction: {analysis['obstruction_ratio'] * 100:.1f}%  "
                f"Elevated: {analysis['elevated_ratio'] * 100:.1f}%  "
                f"Vehicle: {analysis['vehicle_coverage'] * 100:.1f}%",
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # --- Panel 2: Depth colormap ---
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    cv2.polylines(depth_colored, [poly_np], True, (255, 255, 255), 2)

    # --- Panel 3: Mask breakdown ---
    mask_vis = np.zeros_like(frame)
    # Green = vehicle masks (explained)
    mask_vis[analysis["vehicle_mask"] > 0] = (0, 200, 0)
    # Yellow = person masks (excluded from obstruction)
    mask_vis[analysis["person_mask"] > 0] = (0, 255, 255)
    # Red = obstruction (unexplained elevated area)
    mask_vis[analysis["obstruction_mask"]] = (0, 0, 255)
    cv2.polylines(mask_vis, [poly_np], True, (255, 255, 255), 2)
    cv2.putText(mask_vis, "GREEN=Vehicle  YELLOW=Person  RED=Obstruction", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    panel = np.hstack([output, depth_colored, mask_vis])
    return output, panel


def main():
    parser = argparse.ArgumentParser(description="Obstruction detection with YOLO-seg + Depth")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="ai_models/onnx/model_quantized.onnx",
                        help="Depth ONNX model path")
    parser.add_argument("--depth-size", type=int, default=384)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--depth-thresh", type=float, default=0.12,
                        help="Depth deviation threshold for 'elevated'")
    parser.add_argument("--obstruct-thresh", type=float, default=0.05,
                        help="Polygon area fraction for obstruction flag (5%%)")
    parser.add_argument("--no-slots", action="store_true", help="Use full image as zone")
    parser.add_argument("--polygon", type=str, default=None,
                        help="Custom polygon: 'x1,y1;x2,y2;x3,y3;x4,y4'")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERROR: Cannot read image: {args.image}")
        return

    img_h, img_w = frame.shape[:2]
    print(f"Image: {args.image} ({img_w}x{img_h})")

    # Load models
    print("Loading YOLO-seg...")
    yolo = YOLO("yolo11x-seg.pt")

    print("Loading Depth model...")
    import onnxruntime as ort
    depth_session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    # Run YOLO segmentation
    print("\nRunning YOLO-seg...")
    detections, masks_combined, yolo_time = run_yolo_seg(yolo, frame, args.conf)
    cars_total = sum(1 for d in detections if d["class_id"] == 2)
    tw_total = sum(1 for d in detections if d["class_id"] == 3)
    persons_total = sum(1 for d in detections if d["class_id"] == PERSON_ID)
    print(f"  {len(detections)} detections in {yolo_time:.3f}s")
    print(f"  Total: {cars_total} cars, {tw_total} 2-wheelers, {persons_total} persons")

    # Run depth
    print("\nRunning Depth...")
    depth_norm, depth_time = run_depth(frame, depth_session, args.depth_size)
    print(f"  Depth: {depth_time:.3f}s")

    # Define polygon
    if args.polygon:
        polygon = [[int(c) for c in pt.split(",")] for pt in args.polygon.split(";")]
    elif args.no_slots:
        m = 10
        polygon = [[m, m], [img_w - m, m], [img_w - m, img_h - m], [m, img_h - m]]
    else:
        polygon = [[10, 10], [img_w - 10, 10],
                    [img_w - 10, int(img_h * 0.6)], [10, int(img_h * 0.6)]]

    # Analyze
    print(f"\nAnalyzing (depth_thresh={args.depth_thresh}, obstruct_thresh={args.obstruct_thresh})...")
    analysis = detect_obstruction(
        depth_norm, detections, masks_combined, polygon,
        args.depth_thresh, args.obstruct_thresh,
    )

    # Results
    print(f"\n{'=' * 60}")
    print(f"  STATUS:           {'*** OBSTRUCTED ***' if analysis['obstructed'] else 'CLEAR'}")
    print(f"  Cars:             {analysis['cars']}")
    print(f"  2-Wheelers:       {analysis['two_wheelers']}")
    print(f"  Persons:          {analysis['persons']}")
    print(f"  Ground ref:       {analysis['ground_ref']:.1f}/255")
    print(f"  Elevated:         {analysis['elevated_ratio'] * 100:.1f}% of polygon")
    print(f"  Vehicle coverage: {analysis['vehicle_coverage'] * 100:.1f}% of polygon (seg masks)")
    print(f"  Obstruction:      {analysis['obstruction_ratio'] * 100:.1f}% of polygon")
    print(f"  Threshold:        {args.obstruct_thresh * 100:.1f}%")
    print(f"  Timing:           YOLO={yolo_time:.3f}s  Depth={depth_time:.3f}s  Total={yolo_time + depth_time:.3f}s")
    print(f"{'=' * 60}")

    # Save
    os.makedirs("data/debug", exist_ok=True)
    annotated, panel = draw_results(frame, depth_norm, detections, masks_combined, analysis, polygon)
    cv2.imwrite("data/debug/obstruction_seg_result.jpg", annotated)
    cv2.imwrite("data/debug/obstruction_seg_panel.jpg", panel)
    print(f"\nSaved: data/debug/obstruction_seg_result.jpg")
    print(f"Saved: data/debug/obstruction_seg_panel.jpg")
    print(f"  Panel: [seg overlay + annotations | depth | mask breakdown]")
    print(f"  Mask: GREEN=vehicles  YELLOW=persons  RED=obstruction")


if __name__ == "__main__":
    main()
