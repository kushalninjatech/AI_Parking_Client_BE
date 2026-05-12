"""Depth estimation using Depth-Anything-V2-Small ONNX model.
Ported from R&D/poc/depth_estimator.py
"""

import logging
from typing import Optional

import cv2
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DepthEstimator:
    """Depth-Anything-V2-Small. Loads once, shared across all cameras."""

    def __init__(self) -> None:
        self._session = None
        self._loaded = False
        self._input_size = settings.DEPTH_INPUT_SIZE

    def load(self) -> bool:
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                settings.DEPTH_MODEL_PATH,
                providers=["CPUExecutionProvider"],
            )
            self._loaded = True
            logger.info("Depth model loaded from %s (input=%d)", settings.DEPTH_MODEL_PATH, self._input_size)
            return True
        except Exception:
            logger.exception("Failed to load depth model")
            return False

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run depth estimation on full frame.

        Returns depth map as uint8 (0-255) at original frame resolution.
        """
        if not self._loaded:
            return None

        h, w = frame.shape[:2]

        # Preprocess: BGR → RGB, resize, normalize
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._input_size, self._input_size))
        normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
        blob = normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        # Inference
        input_name = self._session.get_inputs()[0].name
        output = self._session.run(None, {input_name: blob})[0]

        # Post-process: squeeze, resize back, normalize to 0-255
        depth = output.squeeze()
        depth = cv2.resize(depth, (w, h))
        depth = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)

        return depth
