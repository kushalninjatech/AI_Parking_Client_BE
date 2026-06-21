"""Test Gemini LLM vision detector on parking images.

Usage:
    # Full image (no polygon)
    python3 test_gemini.py --image raw_frame.jpg --api-key YOUR_KEY

    # With custom polygon
    python3 test_gemini.py --image raw_frame.jpg --api-key YOUR_KEY --polygon "x1,y1;x2,y2;x3,y3;x4,y4"

    # Custom model
    python3 test_gemini.py --image raw_frame.jpg --api-key YOUR_KEY --model gemini-2.5-flash

    # Multiple runs for consistency check
    python3 test_gemini.py --image raw_frame.jpg --api-key YOUR_KEY --runs 3

Install: pip install google-genai opencv-python
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

PROMPT = """Analyze this parking area image. Count all parked/stationary vehicles visible.

Return ONLY valid JSON (no markdown, no code blocks):
{
  "car": 0,
  "two_wheeler": 0,
  "auto_rickshaw": 0,
  "bus": 0,
  "truck": 0,
  "tempo": 0,
  "is_obstructed": false,
  "obstruction_type": null,
  "confidence": 0.9
}

Rules:
- Count only parked/stationary vehicles, not moving ones.
- "car" includes sedans, SUVs, hatchbacks, jeeps.
- "two_wheeler" includes motorcycles, scooters, bicycles.
- "auto_rickshaw" includes 3-wheeled auto rickshaws.
- "tempo" includes small commercial goods vehicles, mini trucks.
- "is_obstructed": true if any non-vehicle object is blocking parking spots.
- Obstructions: street vendor carts, construction material, debris, barricades, fallen objects, encroachments — anything that is NOT a vehicle but blocks parking space.
- "obstruction_type": describe what is obstructing (e.g. "street_vendor_cart", "construction_material", "debris") or null if no obstruction.
- "confidence": your confidence in the analysis (0.0 to 1.0).
- If the image is too dark or unclear, set confidence below 0.5."""


def crop_roi(frame, polygon):
    """Crop frame to polygon bounding box with 10% padding."""
    h, w = frame.shape[:2]
    pts = np.array(polygon, dtype=np.int32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    pad_x = int((x2 - x1) * 0.1)
    pad_y = int((y2 - y1) * 0.1)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return frame[y1:y2, x1:x2]


def call_gemini(client, model_name, image_bytes, run_num=1):
    """Send image to Gemini and return parsed result + timing."""
    from google.genai import types

    t0 = time.monotonic()
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            PROMPT,
        ],
    )
    elapsed = time.monotonic() - t0

    raw_text = response.text.strip()

    # Parse JSON (strip code fences if present)
    cleaned = raw_text
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"  [Run {run_num}] PARSE ERROR: {raw_text[:200]}")
        return None, elapsed, raw_text

    result = {
        "car": int(data.get("car", 0)),
        "two_wheeler": int(data.get("two_wheeler", 0)),
        "auto_rickshaw": int(data.get("auto_rickshaw", 0)),
        "bus": int(data.get("bus", 0)),
        "truck": int(data.get("truck", 0)),
        "tempo": int(data.get("tempo", 0)),
        "is_obstructed": bool(data.get("is_obstructed", False)),
        "obstruction_type": data.get("obstruction_type"),
        "confidence": float(data.get("confidence", 0.5)),
    }
    return result, elapsed, raw_text


def draw_result(frame, result, title):
    """Annotate frame with Gemini detection results."""
    output = frame.copy()
    y = 30
    color = (0, 0, 255) if result.get("is_obstructed") else (0, 200, 0)

    cv2.putText(output, title, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y += 35

    status = "OBSTRUCTED" if result.get("is_obstructed") else "CLEAR"
    cv2.putText(output, status, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    y += 35

    lines = [
        f"Cars: {result['car']}  2W: {result['two_wheeler']}  Auto: {result['auto_rickshaw']}",
        f"Bus: {result['bus']}  Truck: {result['truck']}  Tempo: {result['tempo']}",
        f"Confidence: {result['confidence']:.2f}",
    ]
    if result.get("obstruction_type"):
        lines.append(f"Obstruction: {result['obstruction_type']}")

    for line in lines:
        cv2.putText(output, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 28

    return output


def main():
    parser = argparse.ArgumentParser(description="Test Gemini vision detector")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--api-key", type=str, required=True, help="Gemini API key")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--polygon", type=str, default=None,
                        help="Custom polygon: 'x1,y1;x2,y2;x3,y3;x4,y4'")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs for consistency")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERROR: Cannot read image: {args.image}")
        return

    img_h, img_w = frame.shape[:2]
    print(f"Image: {args.image} ({img_w}x{img_h})")
    print(f"Model: {args.model}")

    # Crop if polygon provided
    if args.polygon:
        polygon = [[int(c) for c in pt.split(",")] for pt in args.polygon.split(";")]
        roi = crop_roi(frame, polygon)
        print(f"Polygon: {polygon}")
        print(f"ROI size: {roi.shape[1]}x{roi.shape[0]}")
    else:
        roi = frame
        print("Using full image (no polygon)")

    # Encode to JPEG
    _, jpeg_bytes = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_bytes = jpeg_bytes.tobytes()
    print(f"JPEG size: {len(image_bytes) / 1024:.1f} KB")

    # Init Gemini
    from google import genai
    client = genai.Client(api_key=args.api_key)
    print("Gemini client initialized\n")

    os.makedirs("data/debug", exist_ok=True)
    all_results = []

    for i in range(args.runs):
        print(f"--- Run {i + 1}/{args.runs} ---")
        result, elapsed, raw_text = call_gemini(client, args.model, image_bytes, i + 1)

        if result is None:
            print(f"  Failed to parse response")
            continue

        total_vehicles = result["car"] + result["two_wheeler"] + result["auto_rickshaw"] + result["bus"] + result["truck"] + result["tempo"]
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Vehicles: {total_vehicles} total")
        print(f"    Cars: {result['car']}  2W: {result['two_wheeler']}  Auto: {result['auto_rickshaw']}")
        print(f"    Bus: {result['bus']}  Truck: {result['truck']}  Tempo: {result['tempo']}")
        print(f"  Obstructed: {result['is_obstructed']}", end="")
        if result["obstruction_type"]:
            print(f" ({result['obstruction_type']})", end="")
        print(f"\n  Confidence: {result['confidence']:.2f}")
        all_results.append({**result, "time": elapsed})

    # Save annotated image (last result)
    if all_results:
        last = all_results[-1]
        title = f"Gemini {args.model} | {last['time']:.2f}s"
        annotated = draw_result(roi, last, title)
        cv2.imwrite("data/debug/gemini_result.jpg", annotated)
        print(f"\nSaved: data/debug/gemini_result.jpg")

    # Consistency summary for multiple runs
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print(f"{'RUN':>4} {'CARS':>5} {'2W':>4} {'AUTO':>5} {'BUS':>4} {'TRUCK':>6} {'TEMPO':>6} {'OBST':>5} {'CONF':>5} {'TIME':>6}")
        print(f"{'=' * 60}")
        for i, r in enumerate(all_results):
            print(f"{i + 1:>4} {r['car']:>5} {r['two_wheeler']:>4} {r['auto_rickshaw']:>5} "
                  f"{r['bus']:>4} {r['truck']:>6} {r['tempo']:>6} "
                  f"{'YES' if r['is_obstructed'] else 'NO':>5} {r['confidence']:>5.2f} {r['time']:>5.2f}s")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
