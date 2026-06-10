"""Test vehicle tracker — runs YOLO + centroid tracking on live camera.

Shows entry/exit events, tracked vehicle counts, and debounce behavior.

Usage (on RPi):
    python3 test_vehicle_tracker.py
    python3 test_vehicle_tracker.py --cycles 10   # run 10 detection cycles

Output:
    data/debug/tracker_frame_N.jpg — annotated frames with tracked vehicles
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from app.core.config import settings
from app.core.constants import COCO_VEHICLE_MAP
from app.detection.yolo_detector import YOLODetector
from app.detection.vehicle_tracker import VehicleTracker
from app.db.session import SessionLocal
from app.models.camera import Camera
from app.models.parking_slot import ParkingSlot
from app.camera.factory import create_camera

COLORS = {
    "CAR": (0, 0, 255),
    "TWO_WHEELER": (255, 100, 0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=5, help="Number of detection cycles")
    parser.add_argument("--interval", type=int, default=10, help="Seconds between cycles")
    args = parser.parse_args()

    print(f"YOLO model: {settings.YOLO_MODEL_PATH}")
    print(f"YOLO input size: {settings.YOLO_INPUT_SIZE}")
    print(f"Cycles: {args.cycles}, Interval: {args.interval}s")

    # Load YOLO
    yolo = YOLODetector()
    if not yolo.load():
        print("ERROR: Failed to load YOLO")
        exit(1)

    # Get camera + slots
    db = SessionLocal()
    cam = db.query(Camera).filter(Camera.is_active == True).first()
    if not cam:
        print("ERROR: No active camera")
        exit(1)

    slots = db.query(ParkingSlot).filter(ParkingSlot.camera_id == cam.id).all()
    print(f"\nCamera: {cam.label} ({cam.source})")
    print(f"Slots: {len(slots)}")

    # Find multi-capacity slots
    trackers = {}
    slot_polygons = {}
    for slot in slots:
        coords = slot.polygon_coords
        if isinstance(coords, str):
            coords = json.loads(coords)
        if not coords:
            continue
        cap = (slot.capacity_car or 0) + (slot.capacity_two_wheeler or 0)
        slot_polygons[slot.id] = {
            "coords": coords,
            "label": slot.label,
            "capacity": cap,
            "np": np.array(coords, dtype=np.int32),
        }
        if cap > 1:
            trackers[slot.id] = VehicleTracker(slot.id, slot.label)
            print(f"  Multi-capacity: {slot.label} (cap={cap}) → using tracker")
        else:
            print(f"  Single-capacity: {slot.label} (cap={cap}) → slot-level")
    db.close()

    if not trackers:
        print("\nNo multi-capacity slots found. Nothing to track.")
        exit(0)

    os.makedirs("data/debug", exist_ok=True)

    for cycle in range(args.cycles):
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle + 1}/{args.cycles}")
        print(f"{'='*60}")

        # Capture frame
        camera = create_camera(cam.source, cam.camera_type)
        if not camera.open():
            print("ERROR: Failed to open camera")
            continue
        frame = None
        for _ in range(5):
            ok, frame = camera.read()
            if ok and frame is not None:
                break
        camera.release()

        if frame is None:
            print("ERROR: No frame captured")
            continue

        img_h, img_w = frame.shape[:2]

        # YOLO detect
        detections = yolo.detect(frame, camera_label=cam.label)
        print(f"YOLO: {len(detections)} vehicles detected")

        # For each tracker slot, filter detections inside polygon
        output = frame.copy()

        for slot_id, tracker in trackers.items():
            sp = slot_polygons[slot_id]
            polygon_np = sp["np"]

            # Filter detections inside polygon
            matched = []
            for det in detections:
                cx, cy = det["centroid"]
                if cv2.pointPolygonTest(polygon_np, (float(cx), float(cy)), False) >= 0:
                    vtype = COCO_VEHICLE_MAP.get(det["class_id"])
                    if vtype:
                        matched.append({**det, "vehicle_type": vtype})

            print(f"\nZone {sp['label']}: {len(matched)} vehicles inside polygon")

            # Update tracker
            result = tracker.update(matched, (img_h, img_w))

            # Print events
            for tv in result["entered"]:
                print(f"  >>> ENTERED: {tv.vehicle_type.value} track={tv.track_id} "
                      f"conf={tv.confidence:.2f} centroid=({tv.centroid[0]},{tv.centroid[1]})")
            for tv in result["exited"]:
                dur = tv.last_seen - tv.first_seen
                print(f"  <<< EXITED:  {tv.vehicle_type.value} track={tv.track_id} "
                      f"duration={dur:.0f}s")

            print(f"  Occupied: {result['occupied_car']} cars, {result['occupied_two_wheeler']} 2W")
            print(f"  Tracked: {tracker.confirmed_count} confirmed")

            # Draw on output
            cv2.polylines(output, [polygon_np], True, (0, 255, 0), 2)

            # Draw tracked vehicles
            for tv in tracker.tracked_vehicles:
                color = COLORS.get(tv.vehicle_type.value, (0, 255, 0))
                x1, y1, x2, y2 = tv.bbox
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                cx, cy = tv.centroid
                cv2.circle(output, (cx, cy), 8, color, -1)
                status = "OK" if tv.confirmed else f"P({tv.seen_count}/3)"
                label = f"{tv.track_id} {tv.vehicle_type.value} {status}"
                cv2.putText(output, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw all YOLO detections (dimmed)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(output, (x1, y1), (x2, y2), (128, 128, 128), 1)

        # Save
        info = f"Cycle {cycle+1}/{args.cycles}"
        cv2.putText(output, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        path = f"data/debug/tracker_frame_{cycle+1}.jpg"
        cv2.imwrite(path, output)
        print(f"\nSaved: {path}")

        if cycle < args.cycles - 1:
            print(f"Waiting {args.interval}s...")
            time.sleep(args.interval)

    print(f"\n{'='*60}")
    print("DONE. Check data/debug/tracker_frame_*.jpg")


if __name__ == "__main__":
    main()
