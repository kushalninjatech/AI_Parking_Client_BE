"""Test Depth-Anything-V2 model — visualize depth map with color overlays.

Usage:
    python3 test_depth.py --image raw_frame.jpg
    python3 test_depth.py --image raw_frame.jpg --model ai_models/onnx/model_quantized.onnx
    python3 test_depth.py --image raw_frame.jpg --size 518

Install: pip install onnxruntime opencv-python numpy
"""

import argparse
import os
import time

import cv2
import numpy as np

# ImageNet normalization (same as depth_estimator.py)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def run_depth(frame, session, input_size, num_runs=3):
    """Run depth estimation and return depth map + timing."""
    h, w = frame.shape[:2]

    # Preprocess: BGR -> RGB, resize, normalize
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size))
    normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    blob = normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    input_name = session.get_inputs()[0].name

    # Warmup
    session.run(None, {input_name: blob})

    # Timed runs
    times = []
    output = None
    for _ in range(num_runs):
        t0 = time.monotonic()
        output = session.run(None, {input_name: blob})[0]
        times.append(time.monotonic() - t0)

    avg_t = sum(times) / len(times)

    # Post-process: squeeze, resize back, normalize to 0-255
    depth = output.squeeze()
    depth = cv2.resize(depth, (w, h))
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)

    return depth_norm, depth, avg_t


def main():
    parser = argparse.ArgumentParser(description="Test Depth-Anything-V2 visualization")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--model", type=str, default="ai_models/onnx/model_quantized.onnx",
                        help="ONNX model path")
    parser.add_argument("--size", type=int, default=384, help="Input size (default: 384)")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERROR: Cannot read image: {args.image}")
        return

    img_h, img_w = frame.shape[:2]
    print(f"Image: {args.image} ({img_w}x{img_h})")
    print(f"Model: {args.model}")
    print(f"Input size: {args.size}")

    # Load ONNX model
    import onnxruntime as ort
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    print(f"Model loaded. Input: {session.get_inputs()[0].shape}")

    # Run depth estimation
    depth_norm, depth_raw, avg_t = run_depth(frame, session, args.size, args.runs)
    print(f"\nDepth inference: avg={avg_t:.3f}s ({args.runs} runs)")
    print(f"Depth range: min={depth_raw.min():.2f} max={depth_raw.max():.2f} mean={depth_raw.mean():.2f}")

    os.makedirs("data/debug", exist_ok=True)

    # 1. Grayscale depth map
    cv2.imwrite("data/debug/depth_gray.jpg", depth_norm)
    print("Saved: data/debug/depth_gray.jpg")

    # 2. Color-mapped depth (INFERNO — warm=close, cool=far)
    depth_inferno = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    cv2.imwrite("data/debug/depth_inferno.jpg", depth_inferno)
    print("Saved: data/debug/depth_inferno.jpg")

    # 3. Color-mapped depth (JET — red=close, blue=far)
    depth_jet = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    cv2.imwrite("data/debug/depth_jet.jpg", depth_jet)
    print("Saved: data/debug/depth_jet.jpg")

    # 4. Color-mapped depth (TURBO — better perceptual uniformity)
    depth_turbo = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    cv2.imwrite("data/debug/depth_turbo.jpg", depth_turbo)
    print("Saved: data/debug/depth_turbo.jpg")

    # 5. Overlay: depth blended on original image (50% opacity)
    overlay = cv2.addWeighted(frame, 0.5, depth_turbo, 0.5, 0)
    cv2.imwrite("data/debug/depth_overlay.jpg", overlay)
    print("Saved: data/debug/depth_overlay.jpg")

    # 6. Side-by-side: original | depth colored
    side_by_side = np.hstack([frame, depth_turbo])
    cv2.imwrite("data/debug/depth_sidebyside.jpg", side_by_side)
    print("Saved: data/debug/depth_sidebyside.jpg")

    # 7. Depth zones visualization (near/mid/far)
    zones = np.zeros_like(frame)
    zones[depth_norm > 170] = (0, 0, 255)    # near = red (high depth value)
    zones[(depth_norm > 85) & (depth_norm <= 170)] = (0, 255, 255)  # mid = yellow
    zones[depth_norm <= 85] = (255, 0, 0)     # far = blue
    zones_overlay = cv2.addWeighted(frame, 0.6, zones, 0.4, 0)
    cv2.putText(zones_overlay, "RED=near  YELLOW=mid  BLUE=far", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite("data/debug/depth_zones.jpg", zones_overlay)
    print("Saved: data/debug/depth_zones.jpg")

    print(f"\nAll outputs saved to data/debug/")


if __name__ == "__main__":
    main()
