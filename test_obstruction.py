"""Test obstruction detection: YOLO + Depth subtraction approach.

Pipeline:
  1. Run YOLO → detect cars, 2W, persons (known objects)
  2. Run Depth → get full depth map
  3. For each slot polygon:
     a. Get depth ROI within polygon
     b. Compare to calibrated ground reference (or use ground plane estimation)
     c. Find "elevated" pixels (something is there)
     d. Subtract YOLO bbox regions (known vehicles/persons)
     e. Remaining elevated area = potential obstruction
  4. If obstruction_ratio > threshold → OBSTRUCTED

Usage:
    # With slot polygons from DB (on RPi)
    python3 test_obstruction.py --image raw_frame.jpg

    # Without slots — analyze full image
    python3 test_obstruction.py --image raw_frame.jpg --no-slots

    # Custom thresholds
    python3 test_obstruction.py --image raw_frame.jpg --depth-thresh 0.15 --obstruct-thresh 0.05

Install: pip install ultralytics onnxruntime opencv-python numpy
"""

import argparse
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

# COCO classes
VEHICLE_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
KNOWN_VEHICLE = {2, 3, 5, 7}  # car, motorcycle, bus, truck
PERSON_CLASS = 0

# ImageNet normalization for depth model
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COLORS = {
    "car": (0, 0, 255),
    "motorcycle": (255, 100, 0),
    "person": (0, 255, 0),
    "bus": (0, 180, 0),
    "truck": (180, 0, 180),
    "obstruction": (0, 0, 255),  # bright red fill
}


def run_yolo(model, frame, conf=0.25):
    """Run YOLO, return detections."""
    t0 = time.monotonic()
    results = model(frame, imgsz=640, conf=conf, verbose=False)[0]
    elapsed = time.monotonic() - t0

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue
        c = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detections.append({
            "class_id": cls_id,
            "class_name": VEHICLE_CLASSES[cls_id],
            "confidence": c,
            "bbox": [x1, y1, x2, y2],
        })
    return detections, elapsed


def run_depth(frame, session, input_size=384):
    """Run depth estimation, return normalized depth map (0-255 uint8)."""
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


def estimate_ground_plane(depth_norm, frame_h):
    """Estimate ground depth by sampling the bottom-center strip of the frame.
    Assumes camera is elevated, looking down — bottom of frame = closest ground.
    Returns per-row expected ground depth (linear gradient).
    """
    # Sample ground from bottom 20% center strip (likely empty pavement)
    strip_top = int(frame_h * 0.8)
    strip_h = frame_h - strip_top

    # Use median of bottom strip as "near ground" reference
    bottom_strip = depth_norm[strip_top:, :]
    near_ground = float(np.median(bottom_strip))

    # Use median of top 20% as "far ground" reference
    top_strip = depth_norm[:int(frame_h * 0.2), :]
    far_ground = float(np.median(top_strip))

    # Linear gradient from far (top) to near (bottom)
    ground_ref = np.linspace(far_ground, near_ground, frame_h).astype(np.float32)
    return ground_ref, near_ground, far_ground


def detect_obstruction(depth_norm, detections, polygon, depth_thresh, obstruct_thresh):
    """Detect obstruction within a polygon region.

    Args:
        depth_norm: uint8 depth map (0-255), higher = closer
        detections: YOLO detections with bboxes
        polygon: list of [x,y] points defining the slot area
        depth_thresh: normalized threshold for "elevated above ground" (0-1)
        obstruct_thresh: fraction of polygon area that must be obstructed

    Returns:
        dict with obstruction analysis results
    """
    h, w = depth_norm.shape[:2]
    poly_np = np.array(polygon, dtype=np.int32)

    # Create polygon mask
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [poly_np], 255)
    poly_area = cv2.countNonZero(poly_mask)

    if poly_area == 0:
        return {"obstructed": False, "reason": "empty polygon"}

    # Build per-row ground plane reference within polygon.
    # Camera looks down at an angle — ground depth increases linearly from top (far) to bottom (near).
    # Sample ground depth at multiple horizontal strips EXCLUDING YOLO detections,
    # then interpolate a smooth gradient. Pixels deviating from this gradient = objects/obstructions.
    px, py, pw, ph = cv2.boundingRect(poly_np)

    # Pre-build YOLO mask to exclude known objects from ground sampling
    yolo_sample_mask = np.zeros((h, w), dtype=np.uint8)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        bw, bh = x2 - x1, y2 - y1
        cv2.rectangle(yolo_sample_mask, (max(0, x1 - int(bw*0.15)), max(0, y1 - int(bh*0.15))),
                      (min(w, x2 + int(bw*0.15)), min(h, y2 + int(bh*0.15))), 255, -1)

    # Sample ground depth in N horizontal strips across polygon (excluding YOLO objects).
    # Use 25th percentile (not median) — ground is the LOWEST depth at each row.
    # Objects/obstructions sit ABOVE ground, so they have higher depth deviation.
    # Using p25 ensures we fit the actual ground surface, not objects on top of it.
    num_strips = 20
    strip_rows = []
    strip_depths = []
    for i in range(num_strips):
        row_y = py + int(ph * i / num_strips)
        row_y_end = min(row_y + max(ph // num_strips, 3), py + ph)
        strip = poly_mask[row_y:row_y_end, :].copy()
        strip[yolo_sample_mask[row_y:row_y_end, :] > 0] = 0  # exclude YOLO regions
        pixels = depth_norm[row_y:row_y_end, :][strip > 0]
        if len(pixels) > 5:
            strip_rows.append((row_y + row_y_end) // 2)
            # p25 = ground surface (lowest depth layer, ignoring objects above)
            strip_depths.append(float(np.percentile(pixels, 25)))

    # Fit linear ground plane: depth = a * row + b
    if len(strip_rows) >= 3:
        coeffs = np.polyfit(strip_rows, strip_depths, 1)
        ground_plane = np.polyval(coeffs, np.arange(h)).astype(np.float32)
    elif len(strip_rows) >= 1:
        # Not enough points — use constant
        ground_plane = np.full(h, np.mean(strip_depths), dtype=np.float32)
    else:
        # Fallback: estimate from full frame
        ground_ref_arr, near, far = estimate_ground_plane(depth_norm, h)
        ground_plane = ground_ref_arr

    ground_ref = float(np.mean(ground_plane[py:py+ph]))

    # Build 2D ground reference (broadcast per-row values across width)
    ground_2d = np.repeat(ground_plane[:, np.newaxis], w, axis=1)

    # Find elevated pixels: deviation from expected ground plane at each pixel
    depth_float = depth_norm.astype(np.float32)
    deviation = np.abs(depth_float - ground_2d) / 255.0
    elevated_mask = (deviation > depth_thresh) & (poly_mask > 0)
    elevated_area = cv2.countNonZero(elevated_mask.astype(np.uint8))

    # Create YOLO known-object mask (vehicles + persons inside polygon)
    yolo_mask = np.zeros((h, w), dtype=np.uint8)
    cars = 0
    two_wheelers = 0
    persons = 0

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # Check if detection centroid is inside polygon
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if cv2.pointPolygonTest(poly_np.astype(np.float32), (float(cx), float(cy)), False) >= 0:
            # Expand bbox slightly (10%) to cover edges
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.1), int(bh * 0.1)
            bx1 = max(0, x1 - pad_x)
            by1 = max(0, y1 - pad_y)
            bx2 = min(w, x2 + pad_x)
            by2 = min(h, y2 + pad_y)
            cv2.rectangle(yolo_mask, (bx1, by1), (bx2, by2), 255, -1)

            if det["class_id"] == 2:
                cars += 1
            elif det["class_id"] == 3:
                two_wheelers += 1
            elif det["class_id"] == PERSON_CLASS:
                persons += 1

    yolo_area = cv2.countNonZero(yolo_mask & poly_mask)

    # Obstruction = elevated area NOT explained by YOLO
    obstruction_mask = elevated_mask & (yolo_mask == 0) & (poly_mask > 0)
    obstruction_area = cv2.countNonZero(obstruction_mask.astype(np.uint8))

    obstruction_ratio = obstruction_area / poly_area if poly_area > 0 else 0
    elevated_ratio = elevated_area / poly_area if poly_area > 0 else 0
    yolo_coverage = yolo_area / poly_area if poly_area > 0 else 0

    is_obstructed = obstruction_ratio > obstruct_thresh

    return {
        "obstructed": is_obstructed,
        "obstruction_ratio": obstruction_ratio,
        "elevated_ratio": elevated_ratio,
        "yolo_coverage": yolo_coverage,
        "obstruction_area": obstruction_area,
        "elevated_area": elevated_area,
        "yolo_area": yolo_area,
        "poly_area": poly_area,
        "ground_ref": ground_ref,
        "cars": cars,
        "two_wheelers": two_wheelers,
        "persons": persons,
        "obstruction_mask": obstruction_mask,
    }


def draw_results(frame, depth_norm, detections, analysis, polygon):
    """Draw comprehensive visualization."""
    h, w = frame.shape[:2]
    output = frame.copy()
    poly_np = np.array(polygon, dtype=np.int32)

    # Draw slot polygon
    color = (0, 0, 255) if analysis["obstructed"] else (0, 255, 0)
    cv2.polylines(output, [poly_np], True, color, 3)

    # Draw YOLO detections
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if cv2.pointPolygonTest(poly_np.astype(np.float32), (float(cx), float(cy)), False) >= 0:
            det_color = COLORS.get(det["class_name"], (255, 255, 255))
            cv2.rectangle(output, (x1, y1), (x2, y2), det_color, 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(output, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, det_color, 2)

    # Overlay obstruction areas in red
    if analysis["obstructed"]:
        obstruct_overlay = output.copy()
        obstruct_overlay[analysis["obstruction_mask"]] = (0, 0, 255)
        output = cv2.addWeighted(output, 0.6, obstruct_overlay, 0.4, 0)

    # Status text
    status = "OBSTRUCTED" if analysis["obstructed"] else "CLEAR"
    status_color = (0, 0, 255) if analysis["obstructed"] else (0, 200, 0)
    cv2.putText(output, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    cv2.putText(output, f"Cars: {analysis['cars']}  2W: {analysis['two_wheelers']}  Persons: {analysis['persons']}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(output, f"Obstruction: {analysis['obstruction_ratio']*100:.1f}%  "
                f"Elevated: {analysis['elevated_ratio']*100:.1f}%  "
                f"YOLO coverage: {analysis['yolo_coverage']*100:.1f}%",
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Build debug panel: original | depth colored | obstruction mask
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    cv2.polylines(depth_colored, [poly_np], True, (255, 255, 255), 2)

    obstruct_vis = np.zeros_like(frame)
    obstruct_vis[analysis["obstruction_mask"]] = (0, 0, 255)  # red = obstruction
    # Show YOLO regions in green
    yolo_mask_vis = np.zeros((h, w), dtype=np.uint8)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if cv2.pointPolygonTest(poly_np.astype(np.float32), (float(cx), float(cy)), False) >= 0:
            cv2.rectangle(obstruct_vis, (x1, y1), (x2, y2), (0, 255, 0), -1)
    cv2.polylines(obstruct_vis, [poly_np], True, (255, 255, 255), 2)
    cv2.putText(obstruct_vis, "GREEN=YOLO  RED=Obstruction", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    panel = np.hstack([output, depth_colored, obstruct_vis])
    return output, panel


def main():
    parser = argparse.ArgumentParser(description="Test obstruction detection")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--model", type=str, default="ai_models/onnx/model_quantized.onnx",
                        help="Depth ONNX model path")
    parser.add_argument("--depth-size", type=int, default=384, help="Depth input size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--depth-thresh", type=float, default=0.12,
                        help="Depth deviation threshold (0-1) for 'elevated' detection")
    parser.add_argument("--obstruct-thresh", type=float, default=0.05,
                        help="Fraction of polygon area for obstruction flag (default: 5%%)")
    parser.add_argument("--no-slots", action="store_true",
                        help="No slot polygons — use full image as one zone")
    parser.add_argument("--polygon", type=str, default=None,
                        help="Custom polygon as 'x1,y1;x2,y2;x3,y3;x4,y4'")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERROR: Cannot read image: {args.image}")
        return

    img_h, img_w = frame.shape[:2]
    print(f"Image: {args.image} ({img_w}x{img_h})")

    # --- Load models ---
    print("Loading YOLO...")
    yolo = YOLO("yolo11x.pt")

    print("Loading Depth model...")
    import onnxruntime as ort
    depth_session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    # --- Run inference ---
    print("\nRunning YOLO...")
    detections, yolo_time = run_yolo(yolo, frame, args.conf)
    print(f"  YOLO: {len(detections)} detections in {yolo_time:.3f}s")

    cars_total = sum(1 for d in detections if d["class_id"] == 2)
    tw_total = sum(1 for d in detections if d["class_id"] == 3)
    persons_total = sum(1 for d in detections if d["class_id"] == PERSON_CLASS)
    print(f"  Total: {cars_total} cars, {tw_total} 2-wheelers, {persons_total} persons")

    print("\nRunning Depth...")
    depth_norm, depth_time = run_depth(frame, depth_session, args.depth_size)
    print(f"  Depth: {depth_time:.3f}s")

    # --- Define polygon ---
    if args.polygon:
        polygon = [[int(c) for c in pt.split(",")] for pt in args.polygon.split(";")]
    elif args.no_slots:
        margin = 10
        polygon = [[margin, margin], [img_w - margin, margin],
                    [img_w - margin, img_h - margin], [margin, img_h - margin]]
    else:
        # Default: use top 60% of frame as parking zone (where vehicles typically are)
        polygon = [[10, 10], [img_w - 10, 10],
                    [img_w - 10, int(img_h * 0.6)], [10, int(img_h * 0.6)]]

    print(f"\nPolygon: {polygon}")

    # --- Obstruction analysis ---
    print(f"\nAnalyzing obstruction (depth_thresh={args.depth_thresh}, obstruct_thresh={args.obstruct_thresh})...")
    analysis = detect_obstruction(
        depth_norm, detections, polygon,
        depth_thresh=args.depth_thresh,
        obstruct_thresh=args.obstruct_thresh,
    )

    # --- Results ---
    print(f"\n{'='*60}")
    print(f"  STATUS:        {'*** OBSTRUCTED ***' if analysis['obstructed'] else 'CLEAR'}")
    print(f"  Cars:          {analysis['cars']}")
    print(f"  2-Wheelers:    {analysis['two_wheelers']}")
    print(f"  Persons:       {analysis['persons']}")
    print(f"  Ground ref:    {analysis['ground_ref']:.1f}/255")
    print(f"  Elevated:      {analysis['elevated_ratio']*100:.1f}% of polygon")
    print(f"  YOLO coverage: {analysis['yolo_coverage']*100:.1f}% of polygon")
    print(f"  Obstruction:   {analysis['obstruction_ratio']*100:.1f}% of polygon")
    print(f"  Threshold:     {args.obstruct_thresh*100:.1f}%")
    print(f"{'='*60}")

    # --- Save visualizations ---
    os.makedirs("data/debug", exist_ok=True)

    annotated, panel = draw_results(frame, depth_norm, detections, analysis, polygon)
    cv2.imwrite("data/debug/obstruction_result.jpg", annotated)
    cv2.imwrite("data/debug/obstruction_panel.jpg", panel)
    print(f"\nSaved: data/debug/obstruction_result.jpg")
    print(f"Saved: data/debug/obstruction_panel.jpg")
    print(f"  Panel: [original+annotations | depth colored | obstruction mask]")


if __name__ == "__main__":
    main()
