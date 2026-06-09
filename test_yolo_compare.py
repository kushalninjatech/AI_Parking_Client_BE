"""Compare YOLO models on same image. Runs on Mac using ultralytics.

Tests different models/sizes to find what detects near-camera vehicles best.

Usage:
    # First, SCP a frame from RPi:
    #   scp rpi:~/AI_PARKING_CLIENT/AI_Parking_Client_BE/data/debug/raw_frame.jpg .

    # Compare models
    python3 test_yolo_compare.py --image raw_frame.jpg

    # Test specific model only
    python3 test_yolo_compare.py --image raw_frame.jpg --models yolo11x

    # Custom input size
    python3 test_yolo_compare.py --image raw_frame.jpg --models yolo11x --size 2048

    # Test all models at multiple sizes
    python3 test_yolo_compare.py --image raw_frame.jpg --sizes 640,1280,2048

Install: pip install ultralytics opencv-python
"""

import argparse
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {
    "car": (0, 0, 255),
    "motorcycle": (255, 100, 0),
    "bus": (0, 180, 0),
    "truck": (180, 0, 180),
}

# Models to compare
DEFAULT_MODELS = [
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
]

# NCNN model paths (local directories) — detected by "/" in the name
# For NCNN models, ultralytics loads from the directory directly


def run_model(model_name, frame, input_size, conf):
    """Run a YOLO model and return detections + timing."""
    # Support both .pt names and direct paths (NCNN dirs, .onnx files)
    if os.path.exists(model_name):
        model = YOLO(model_name)
    elif os.path.exists(f"{model_name}.pt"):
        model = YOLO(f"{model_name}.pt")
    else:
        model = YOLO(f"{model_name}.pt")

    t0 = time.monotonic()
    results = model(frame, imgsz=input_size, conf=conf, verbose=False)[0]
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
            "centroid": [(x1 + x2) // 2, (y1 + y2) // 2],
        })

    return detections, elapsed


def draw_result(frame, detections, title):
    """Draw detections on frame with title."""
    output = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = det["centroid"]
        color = COLORS.get(det["class_name"], (0, 0, 255))

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.circle(output, (cx, cy), 5, color, -1)

        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(output, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.putText(output, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return output


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO models")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                        help="Comma-separated model names (e.g. yolo11n,yolo11l,yolo11x)")
    parser.add_argument("--sizes", type=str, default="640,2048",
                        help="Comma-separated input sizes (e.g. 640,1280,2048)")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERROR: Cannot read image: {args.image}")
        return

    img_h, img_w = frame.shape[:2]
    models = [m.strip() for m in args.models.split(",")]
    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    print(f"Image: {args.image} ({img_w}x{img_h})")
    print(f"Models: {models}")
    print(f"Sizes: {sizes}")
    print(f"Confidence: {args.conf}")

    os.makedirs("data/debug", exist_ok=True)
    all_outputs = []

    for model_name in models:
        for input_size in sizes:
            print(f"\n{'='*60}")
            print(f"{model_name} @ {input_size}")
            print(f"{'='*60}")

            detections, elapsed = run_model(model_name, frame, input_size, args.conf)
            print(f"Detected {len(detections)} vehicles in {elapsed:.2f}s\n")

            # Count by region (top half vs bottom half)
            top_half = [d for d in detections if d["centroid"][1] < img_h // 2]
            bottom_half = [d for d in detections if d["centroid"][1] >= img_h // 2]

            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det["bbox"]
                bw, bh = x2 - x1, y2 - y1
                frame_pct = (bw * bh) / (img_w * img_h) * 100
                region = "TOP" if det["centroid"][1] < img_h // 2 else "BOT"
                print(f"  {i+1}. [{region}] {det['class_name']:12s} conf={det['confidence']:.3f}  "
                      f"bbox=({x1},{y1})-({x2},{y2})  size={bw}x{bh}  frame%={frame_pct:.1f}%")

            print(f"\n  Summary: {len(top_half)} top + {len(bottom_half)} bottom = {len(detections)} total")

            # Save individual output
            display_name = os.path.basename(model_name.rstrip("/"))
            title = f"{display_name} @ {input_size} | {len(detections)} det | {elapsed:.2f}s"
            output = draw_result(frame, detections, title)
            out_path = f"data/debug/compare_{display_name}_{input_size}.jpg"
            cv2.imwrite(out_path, output)
            print(f"  Saved: {out_path}")

            all_outputs.append({
                "model": display_name,
                "size": input_size,
                "total": len(detections),
                "top": len(top_half),
                "bottom": len(bottom_half),
                "time": elapsed,
            })

    # Print comparison table
    print(f"\n\n{'='*70}")
    print(f"{'MODEL':<12} {'SIZE':>6} {'TOTAL':>6} {'TOP':>5} {'BOTTOM':>7} {'TIME':>7}")
    print(f"{'='*70}")
    for r in all_outputs:
        print(f"{r['model']:<12} {r['size']:>6} {r['total']:>6} {r['top']:>5} {r['bottom']:>7} {r['time']:>6.2f}s")
    print(f"{'='*70}")
    print(f"\nKey metric: BOTTOM count — higher = better near-camera detection")


if __name__ == "__main__":
    main()
