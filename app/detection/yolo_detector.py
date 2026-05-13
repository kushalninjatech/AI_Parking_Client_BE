"""YOLO vehicle detector using NCNN backend.
Ported from R&D/poc/app_v2.py — runs on full frame, returns all detections.
"""

import logging
from typing import Dict, List

import cv2
import numpy as np

from app.core.config import settings
from app.core.constants import VEHICLE_CLASS_IDS

logger = logging.getLogger(__name__)

YOLO_INPUT_SIZE = 640


class YOLODetector:
    """YOLOv26n NCNN detector. Loads once, shared across all cameras."""

    def __init__(self) -> None:
        self._net = None
        self._loaded = False

    def load(self) -> bool:
        try:
            import ncnn
            self._net = ncnn.Net()
            param_path = f"{settings.YOLO_MODEL_PATH}/model.ncnn.param"
            bin_path = f"{settings.YOLO_MODEL_PATH}/model.ncnn.bin"
            self._net.load_param(param_path)
            self._net.load_model(bin_path)
            self._loaded = True
            logger.info("YOLO model loaded from %s", settings.YOLO_MODEL_PATH)
            return True
        except Exception:
            logger.exception("Failed to load YOLO model")
            return False

    def detect(self, frame: np.ndarray) -> List[Dict]:
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

        import ncnn

        img_h, img_w = frame.shape[:2]

        # Letterbox resize
        scale = min(YOLO_INPUT_SIZE / img_w, YOLO_INPUT_SIZE / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        dx = (YOLO_INPUT_SIZE - new_w) // 2
        dy = (YOLO_INPUT_SIZE - new_h) // 2

        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.full((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[dy:dy + new_h, dx:dx + new_w] = resized

        # NCNN inference
        mat_in = ncnn.Mat.from_pixels(padded, ncnn.Mat.PixelType.PIXEL_BGR, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)
        mean_vals = [0.0, 0.0, 0.0]
        norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        mat_in.substract_mean_normalize(mean_vals, norm_vals)

        ex = self._net.create_extractor()
        ex.input("in0", mat_in)
        _, output = ex.extract("out0")

        # Post-process: NMS
        detections = self._postprocess(output, scale, dx, dy, img_h, img_w)
        return detections

    def _postprocess(
        self, output: "ncnn.Mat", scale: float, dx: int, dy: int, img_h: int, img_w: int
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
            if class_id not in VEHICLE_CLASS_IDS:
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

        # NMS
        indices = cv2.dnn.NMSBoxes(
            [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes],
            scores, settings.YOLO_CONFIDENCE, 0.45,
        )

        results = []
        for i in indices:
            idx = int(np.ravel(i)[0])  # handles int, list[int], and numpy scalar
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
