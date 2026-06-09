"""YOLO vehicle detector — auto-selects backend based on model path.

NCNN backend: YOLO_MODEL_PATH points to a directory with model.ncnn.param + model.ncnn.bin
PT backend:   YOLO_MODEL_PATH points to a .pt file (requires ultralytics)

Set via .env:
  YOLO_MODEL_PATH=ai_models/yolo26l_ncnn_model   # NCNN
  YOLO_MODEL_PATH=ai_models/yolo11x.pt            # PT (ultralytics)
"""

import logging
import os
import time
from typing import Dict, List

import cv2
import numpy as np

from app.core.config import settings
from app.core.constants import COCO_VEHICLE_MAP

logger = logging.getLogger(__name__)

YOLO_INPUT_SIZE = settings.YOLO_INPUT_SIZE


def _is_ncnn_model(path: str) -> bool:
    """Check if path is an NCNN model directory."""
    if os.path.isdir(path):
        return True
    return not path.endswith((".pt", ".onnx"))


class YOLODetector:
    """YOLO detector. Auto-selects NCNN or ultralytics backend from config."""

    def __init__(self) -> None:
        self._backend = None  # "ncnn" or "ultralytics"
        self._net = None      # ncnn.Net for NCNN backend
        self._model = None    # YOLO model for ultralytics backend
        self._loaded = False

    def load(self) -> bool:
        model_path = settings.YOLO_MODEL_PATH

        if _is_ncnn_model(model_path):
            return self._load_ncnn(model_path)
        else:
            return self._load_ultralytics(model_path)

    def _load_ncnn(self, model_path: str) -> bool:
        try:
            import ncnn
            self._net = ncnn.Net()
            param_path = f"{model_path}/model.ncnn.param"
            bin_path = f"{model_path}/model.ncnn.bin"
            self._net.load_param(param_path)
            self._net.load_model(bin_path)
            self._backend = "ncnn"
            self._loaded = True
            logger.info(
                "YOLO NCNN loaded: path=%s input_size=%d confidence=%.2f",
                model_path, YOLO_INPUT_SIZE, settings.YOLO_CONFIDENCE,
            )
            return True
        except Exception:
            logger.exception("Failed to load YOLO NCNN model from %s", model_path)
            return False

    def _load_ultralytics(self, model_path: str) -> bool:
        try:
            from ultralytics import YOLO
            from pathlib import Path

            if not os.path.exists(model_path):
                model_name = os.path.basename(model_path)
                logger.info("Model not found at %s, downloading %s...", model_path, model_name)
                self._model = YOLO(model_name)
                os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
                Path(model_name).rename(model_path)
                logger.info("Model saved to %s", model_path)
            else:
                self._model = YOLO(model_path)

            self._backend = "ultralytics"
            self._loaded = True
            logger.info(
                "YOLO ultralytics loaded: path=%s input_size=%d confidence=%.2f",
                model_path, YOLO_INPUT_SIZE, settings.YOLO_CONFIDENCE,
            )
            return True
        except Exception:
            logger.exception("Failed to load YOLO ultralytics model from %s", model_path)
            return False

    def detect(self, frame: np.ndarray, camera_label: str = "") -> List[Dict]:
        """Run YOLO on full frame. Returns list of vehicle detections.

        Each detection: {
            "class_id": int,
            "confidence": float,
            "bbox": [x1, y1, x2, y2],
            "centroid": [cx, cy],
        }
        """
        if not self._loaded:
            return []

        if self._backend == "ncnn":
            return self._detect_ncnn(frame, camera_label)
        else:
            return self._detect_ultralytics(frame, camera_label)

    # ------------------------------------------------------------------
    # NCNN backend
    # ------------------------------------------------------------------

    def _detect_ncnn(self, frame: np.ndarray, camera_label: str) -> List[Dict]:
        import ncnn

        img_h, img_w = frame.shape[:2]
        t0 = time.monotonic()

        # Letterbox resize
        scale = min(YOLO_INPUT_SIZE / img_w, YOLO_INPUT_SIZE / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        dx = (YOLO_INPUT_SIZE - new_w) // 2
        dy = (YOLO_INPUT_SIZE - new_h) // 2

        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.full((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[dy:dy + new_h, dx:dx + new_w] = resized

        mat_in = ncnn.Mat.from_pixels(padded, ncnn.Mat.PixelType.PIXEL_BGR, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0, 1 / 255.0, 1 / 255.0])

        ex = self._net.create_extractor()
        ex.input("in0", mat_in)
        _, output = ex.extract("out0")

        detections = self._postprocess_ncnn(output, scale, dx, dy, img_h, img_w)
        elapsed = time.monotonic() - t0
        logger.info(
            "YOLO detect [ncnn]: %d vehicles in %.2fs (camera=%s frame=%dx%d input=%d)",
            len(detections), elapsed, camera_label, img_w, img_h, YOLO_INPUT_SIZE,
        )
        return detections

    def _postprocess_ncnn(
        self, output, scale: float, dx: int, dy: int, img_h: int, img_w: int
    ) -> List[Dict]:
        data = np.array(output).reshape(output.h, output.w)
        if data.shape[0] < data.shape[1]:
            data = data.T

        boxes, scores, class_ids = [], [], []

        for row in data:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])

            if conf < settings.YOLO_CONFIDENCE:
                continue
            if class_id not in COCO_VEHICLE_MAP:
                continue

            cx, cy, w, h = row[:4]
            x1 = (cx - w / 2 - dx) / scale
            y1 = (cy - h / 2 - dy) / scale
            x2 = (cx + w / 2 - dx) / scale
            y2 = (cy + h / 2 - dy) / scale

            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes],
            scores, settings.YOLO_CONFIDENCE, 0.45,
        )

        results = []
        for i in indices:
            idx = int(np.ravel(i)[0])
            b = boxes[idx]
            cx = (b[0] + b[2]) / 2
            cy = (b[1] + b[3]) / 2
            results.append({
                "class_id": class_ids[idx],
                "confidence": scores[idx],
                "bbox": [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
                "centroid": [int(cx), int(cy)],
            })

        return results

    # ------------------------------------------------------------------
    # Ultralytics backend
    # ------------------------------------------------------------------

    def _detect_ultralytics(self, frame: np.ndarray, camera_label: str) -> List[Dict]:
        img_h, img_w = frame.shape[:2]
        t0 = time.monotonic()

        results = self._model(
            frame,
            imgsz=YOLO_INPUT_SIZE,
            conf=settings.YOLO_CONFIDENCE,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in COCO_VEHICLE_MAP:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append({
                "class_id": cls_id,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "centroid": [cx, cy],
            })

        elapsed = time.monotonic() - t0
        logger.info(
            "YOLO detect [ultralytics]: %d vehicles in %.2fs (camera=%s frame=%dx%d input=%d)",
            len(detections), elapsed, camera_label, img_w, img_h, YOLO_INPUT_SIZE,
        )
        return detections
