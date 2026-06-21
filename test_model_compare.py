"""Compare YOLOv11x vs RT-DETR vs RF-DETR on same image with timing.

Usage:
    python3 test_model_compare.py --image raw_frame.jpg
    python3 test_model_compare.py --image raw_frame.jpg --models yolo11x,rtdetr-l,rfdetr-b
    python3 test_model_compare.py --image raw_frame.jpg --runs 5

Install:
    pip install ultralytics opencv-python
    pip install rfdetr  # for RF-DETR (Roboflow)
"""

import argparse
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO, RTDETR

# COCO class IDs: vehicles + person
VEHICLE_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# COCO class names (for RF-DETR which returns names)
VEHICLE_NAMES = {"person", "car", "motorcycle", "bus", "truck"}
COLORS = {
    "person": (0, 255, 0),
    "car": (0, 0, 255),
    "motorcycle": (255, 100, 0),
    "bus": (0, 180, 0),
    "truck": (180, 0, 180),
}

DEFAULT_MODELS = ["yolo11x", "rtdetr-l", "rfdetr-b"]
RTDETR_NAMES = {"rtdetr-l", "rtdetr-x"}
RFDETR_NAMES = {"rfdetr-b", "rfdetr-l"}  # base and large

_model_cache = {}


def _get_model_type(model_name):
    """Determine model type from name."""
    base = os.path.basename(model_name.rstrip("/")).replace(".pt", "")
    if base in RFDETR_NAMES:
        return "rfdetr", base
    if base in RTDETR_NAMES:
        return "rtdetr", base
    return "yolo", base


def _load_model(model_name):
    if model_name in _model_cache:
        return _model_cache[model_name]

    mtype, base = _get_model_type(model_name)

    if mtype == "rfdetr":
        from rfdetr import RFDETRBase, RFDETRLarge
        model = RFDETRLarge() if base == "rfdetr-l" else RFDETRBase()
    elif mtype == "rtdetr":
        model = RTDETR(f"{base}.pt")
    elif os.path.exists(model_name):
        model = YOLO(model_name)
    elif os.path.exists(f"{model_name}.pt"):
        model = YOLO(f"{model_name}.pt")
    else:
        model = YOLO(f"{model_name}.pt")

    _model_cache[model_name] = model
    return model


def _iou(box_a, box_b):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _soft_nms(dets, iou_thresh=0.35, sigma=0.5, score_thresh=0.15):
    """Soft-NMS: decay overlapping scores with Gaussian penalty instead of hard removal.
    - iou_thresh: IoU above which scores get decayed (lower = more aggressive)
    - sigma: Gaussian decay width (lower = sharper decay)
    - score_thresh: drop detections below this after decay
    """
    import math
    remaining = list(range(len(dets)))
    kept = []

    while remaining:
        # Pick highest confidence
        best_idx = max(remaining, key=lambda i: dets[i]["confidence"])
        kept.append(best_idx)
        remaining.remove(best_idx)

        best_box = dets[best_idx]["bbox"]
        to_remove = []
        for idx in remaining:
            iou_val = _iou(best_box, dets[idx]["bbox"])
            if iou_val > iou_thresh:
                # Gaussian decay: score *= exp(-iou^2 / sigma)
                decay = math.exp(-(iou_val ** 2) / sigma)
                dets[idx]["confidence"] *= decay
                if dets[idx]["confidence"] < score_thresh:
                    to_remove.append(idx)
        for idx in to_remove:
            remaining.remove(idx)

    return [dets[i] for i in kept]


def _containment_ratio(box_a, box_b):
    """What fraction of box_a's area is inside box_b."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    return inter / area_a if area_a > 0 else 0


def _aggressive_nms(detections, iou_thresh=0.3, containment_thresh=0.6):
    """3-criteria suppression across ALL classes:
    1. IoU > iou_thresh — standard overlap
    2. Containment > containment_thresh — smaller box mostly inside larger box
    3. Centroid distance < 60% of smaller box diagonal — shifted duplicates
    Keeps higher confidence box in all cases.
    """
    if not detections:
        return detections

    import math
    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = []

    for det in dets:
        suppressed = False
        bx = det["bbox"]
        cx, cy = det["centroid"]
        bw = bx[2] - bx[0]
        bh = bx[3] - bx[1]
        diag = math.sqrt(bw**2 + bh**2)

        for kept in keep:
            kx = kept["bbox"]
            kcx, kcy = kept["centroid"]

            # Check 1: IoU overlap
            if _iou(bx, kx) > iou_thresh:
                suppressed = True
                break

            # Check 2: smaller box contained inside larger box
            if _containment_ratio(bx, kx) > containment_thresh:
                suppressed = True
                break
            if _containment_ratio(kx, bx) > containment_thresh:
                # Current box contains the kept box — but kept has higher conf, skip current
                suppressed = True
                break

            # Check 3: centroids very close relative to box size
            cdist = math.sqrt((cx - kcx)**2 + (cy - kcy)**2)
            min_diag = min(diag, math.sqrt((kx[2]-kx[0])**2 + (kx[3]-kx[1])**2))
            if min_diag > 0 and cdist < min_diag * 0.4:
                suppressed = True
                break

        if not suppressed:
            keep.append(det)

    return keep


def _class_aware_soft_nms(detections, iou_thresh=0.3, sigma=0.3, score_thresh=0.15):
    """Two-pass NMS:
    1. Soft-NMS per class (removes same-class duplicates with Gaussian decay)
    2. Aggressive NMS across all classes (IoU + containment + centroid distance)
    """
    # Pass 1: per-class Soft-NMS
    by_class = {}
    for det in detections:
        by_class.setdefault(det["class_id"], []).append(det)

    after_soft = []
    for cls_dets in by_class.values():
        after_soft.extend(_soft_nms(cls_dets, iou_thresh, sigma, score_thresh))

    # Pass 2: aggressive cross-class suppression
    result = _aggressive_nms(after_soft, iou_thresh=0.3, containment_thresh=0.6)

    result.sort(key=lambda d: d["confidence"], reverse=True)
    return result


def _run_rfdetr(model, frame, conf, num_runs):
    """RF-DETR uses supervision.Detections, different API from ultralytics."""
    from PIL import Image

    # RF-DETR expects PIL image
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Warmup
    model.predict(pil_img, threshold=conf)

    times = []
    result = None
    for _ in range(num_runs):
        t0 = time.monotonic()
        result = model.predict(pil_img, threshold=conf)
        times.append(time.monotonic() - t0)

    # result is supervision.Detections with .xyxy, .confidence, .class_id, .data["class_name"]
    detections = []
    for i in range(len(result)):
        cls_id = int(result.class_id[i])
        cls_name = VEHICLE_CLASSES.get(cls_id)
        if not cls_name:
            continue
        c = float(result.confidence[i])
        x1, y1, x2, y2 = [int(v) for v in result.xyxy[i]]
        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": c,
            "bbox": [x1, y1, x2, y2],
            "centroid": [(x1 + x2) // 2, (y1 + y2) // 2],
        })

    return detections, times


def _run_ultralytics(model, frame, input_size, conf, num_runs):
    """YOLO / RT-DETR via ultralytics API."""
    # Warmup
    model(frame, imgsz=input_size, conf=conf, verbose=False)

    times = []
    results = None
    for _ in range(num_runs):
        t0 = time.monotonic()
        results = model(frame, imgsz=input_size, conf=conf, verbose=False)[0]
        times.append(time.monotonic() - t0)

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

    return detections, times


def run_model(model_name, frame, input_size, conf, num_runs=1):
    model = _load_model(model_name)
    mtype, base = _get_model_type(model_name)

    if mtype == "rfdetr":
        detections, times = _run_rfdetr(model, frame, conf, num_runs)
    else:
        detections, times = _run_ultralytics(model, frame, input_size, conf, num_runs)

    avg_t = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)

    # For transformer models (RT-DETR, RF-DETR), apply Soft-NMS to remove duplicates
    import copy
    raw_detections = copy.deepcopy(detections)

    needs_nms = mtype in ("rtdetr", "rfdetr")
    if needs_nms and detections:
        detections = _class_aware_soft_nms(detections, iou_thresh=0.35, sigma=0.5, score_thresh=conf)

    return detections, raw_detections, avg_t, min_t, max_t, needs_nms


def draw_result(frame, detections, title):
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
    parser = argparse.ArgumentParser(description="Compare YOLOv11x vs RT-DETR-L")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                        help="Comma-separated model names (default: yolo11x,rtdetr-l)")
    parser.add_argument("--sizes", type=str, default="640",
                        help="Comma-separated input sizes (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--runs", type=int, default=3, help="Number of timed runs per config (default: 3)")
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
    print(f"Timed runs: {args.runs}")

    os.makedirs("data/debug", exist_ok=True)
    all_results = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Loading {model_name}...")
        print(f"{'='*60}")

        for input_size in sizes:
            print(f"\n--- {model_name} @ {input_size}px ---")

            detections, raw_detections, avg_t, min_t, max_t, needs_nms = run_model(
                model_name, frame, input_size, args.conf, args.runs
            )

            display_name = os.path.basename(model_name.rstrip("/")).replace(".pt", "")

            # For RT-DETR: save raw (no NMS) image first
            if needs_nms:
                raw_cars = [d for d in raw_detections if d["class_name"] == "car"]
                raw_motos = [d for d in raw_detections if d["class_name"] == "motorcycle"]
                print(f"  [RAW - no NMS] {len(raw_detections)} total | "
                      f"{len(raw_cars)} cars, {len(raw_motos)} motorcycles")
                raw_title = f"{display_name}@{input_size} RAW (no NMS) | {len(raw_detections)} det"
                raw_output = draw_result(frame, raw_detections, raw_title)
                raw_path = f"data/debug/compare_{display_name}_{input_size}_raw.jpg"
                cv2.imwrite(raw_path, raw_output)
                print(f"  Saved: {raw_path}")

            # Count by type (after NMS)
            cars = [d for d in detections if d["class_name"] == "car"]
            motos = [d for d in detections if d["class_name"] == "motorcycle"]
            buses = [d for d in detections if d["class_name"] == "bus"]
            trucks = [d for d in detections if d["class_name"] == "truck"]

            nms_label = " [after Soft-NMS]" if needs_nms else ""
            print(f"  {nms_label} {len(detections)} total | "
                  f"{len(cars)} cars, {len(motos)} motorcycles, "
                  f"{len(buses)} buses, {len(trucks)} trucks")
            if needs_nms:
                removed = len(raw_detections) - len(detections)
                print(f"  NMS removed {removed} duplicate boxes")
            print(f"  Timing:   avg={avg_t:.3f}s  min={min_t:.3f}s  max={max_t:.3f}s  "
                  f"({args.runs} runs, warmup excluded)")

            # Print each detection
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det["bbox"]
                bw, bh = x2 - x1, y2 - y1
                frame_pct = (bw * bh) / (img_w * img_h) * 100
                print(f"    {i+1}. {det['class_name']:12s} conf={det['confidence']:.3f}  "
                      f"bbox=({x1},{y1})-({x2},{y2})  size={bw}x{bh}  frame%={frame_pct:.1f}%")

            # Save annotated image (after NMS)
            nms_suffix = "_nms" if needs_nms else ""
            title = f"{display_name}@{input_size}{nms_label} | {len(detections)} det | avg {avg_t:.3f}s"
            output = draw_result(frame, detections, title)
            out_path = f"data/debug/compare_{display_name}_{input_size}{nms_suffix}.jpg"
            cv2.imwrite(out_path, output)
            print(f"  Saved: {out_path}")

            all_results.append({
                "model": display_name,
                "size": input_size,
                "total": len(detections),
                "raw": len(raw_detections) if needs_nms else len(detections),
                "cars": len(cars),
                "motos": len(motos),
                "avg_s": avg_t,
                "min_s": min_t,
                "max_s": max_t,
            })

    # Comparison table
    print(f"\n\n{'='*90}")
    print(f"{'MODEL':<14} {'SIZE':>6} {'RAW':>5} {'NMS':>5} {'CARS':>5} {'MOTOS':>6} {'AVG':>7} {'MIN':>7} {'MAX':>7}")
    print(f"{'='*90}")
    for r in all_results:
        print(f"{r['model']:<14} {r['size']:>6} {r['raw']:>5} {r['total']:>5} {r['cars']:>5} {r['motos']:>6} "
              f"{r['avg_s']:>6.3f}s {r['min_s']:>6.3f}s {r['max_s']:>6.3f}s")
    print(f"{'='*90}")

    # Side-by-side summary
    if len(all_results) >= 2:
        print(f"\n--- Quick Comparison ---")
        a, b = all_results[0], all_results[-1]
        det_diff = b["total"] - a["total"]
        speed_ratio = b["avg_s"] / a["avg_s"] if a["avg_s"] > 0 else 0
        print(f"  {a['model']}@{a['size']}: {a['total']} detections, {a['avg_s']:.3f}s avg")
        print(f"  {b['model']}@{b['size']}: {b['total']} detections, {b['avg_s']:.3f}s avg")
        print(f"  Detection diff: {det_diff:+d} | Speed ratio: {speed_ratio:.2f}x")


if __name__ == "__main__":
    main()
