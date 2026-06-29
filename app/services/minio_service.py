"""MinIO service — uploads cropped slot images to S3-compatible object storage."""

import io
import logging
import time
from typing import Optional

import cv2
import numpy as np
from minio import Minio

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: Optional[Minio] = None


def get_minio_client() -> Optional[Minio]:
    global _client
    if _client is None:
        if not settings.MINIO_ACCESS_KEY or not settings.MINIO_SECRET_KEY:
            logger.warning("MinIO credentials not configured — image upload disabled")
            return None
        _client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
    return _client


def upload_slot_image(
    frame: np.ndarray,
    polygon_coords: list,
    device_id: str,
    camera_label: str,
    slot_label: str,
) -> Optional[str]:
    """Crop slot region from frame and upload to MinIO.

    Returns the public URL or None on failure.
    """
    client = get_minio_client()
    if client is None:
        return None

    try:
        # Crop slot ROI using polygon bounding box
        pts = np.array(polygon_coords, dtype=np.int32)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid crop bounds for slot %s", slot_label)
            return None

        cropped = frame[y1:y2, x1:x2]

        # Encode as JPEG
        ok, buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            logger.warning("Failed to encode slot image for %s", slot_label)
            return None

        data = buf.tobytes()
        ts = int(time.time())
        object_name = f"detections/{device_id}/{camera_label}/{slot_label}/{ts}.jpg"

        client.put_object(
            settings.MINIO_BUCKET,
            object_name,
            io.BytesIO(data),
            len(data),
            content_type="image/jpeg",
        )

        scheme = "https" if settings.MINIO_SECURE else "http"
        url = f"{scheme}://{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET}/{object_name}"
        return url

    except Exception:
        logger.exception("Failed to upload slot image for %s", slot_label)
        return None


def upload_debug_frame(
    frame: np.ndarray,
    device_id: str,
    camera_label: str,
) -> Optional[str]:
    """Upload annotated detection frame to MinIO. Overwrites previous frame."""
    client = get_minio_client()
    if client is None:
        return None

    try:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return None

        data = buf.tobytes()
        object_name = f"debug/{device_id}/{camera_label}/latest.jpg"

        client.put_object(
            settings.MINIO_BUCKET,
            object_name,
            io.BytesIO(data),
            len(data),
            content_type="image/jpeg",
        )

        scheme = "https" if settings.MINIO_SECURE else "http"
        url = f"{scheme}://{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET}/{object_name}"
        logger.debug("Debug frame uploaded: %s", url)
        return url

    except Exception:
        logger.debug("Failed to upload debug frame")
        return None


def upload_clean_frame(
    frame: np.ndarray,
    device_id: str,
    camera_label: str,
) -> Optional[str]:
    """Upload clean frame (polygon borders only) to MinIO for public view."""
    client = get_minio_client()
    if client is None:
        return None

    try:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return None

        data = buf.tobytes()
        object_name = f"debug/{device_id}/{camera_label}/clean.jpg"

        client.put_object(
            settings.MINIO_BUCKET,
            object_name,
            io.BytesIO(data),
            len(data),
            content_type="image/jpeg",
        )

        scheme = "https" if settings.MINIO_SECURE else "http"
        url = f"{scheme}://{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET}/{object_name}"
        return url


def upload_scan_frame(
    frame: np.ndarray,
    device_id: str,
    camera_label: str,
) -> Optional[str]:
    """Upload a timestamped scan frame to MinIO for parking history.

    Unlike clean.jpg (overwritten every cycle), each scan gets a unique file
    so historical images are preserved for PDF export and review.
    Path: scans/{device_id}/{camera_label}/{timestamp}.jpg
    """
    import time
    client = get_minio_client()
    if client is None:
        return None

    try:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return None

        data = buf.tobytes()
        ts = int(time.time())
        object_name = f"scans/{device_id}/{camera_label}/{ts}.jpg"

        client.put_object(
            settings.MINIO_BUCKET,
            object_name,
            io.BytesIO(data),
            len(data),
            content_type="image/jpeg",
        )

        scheme = "https" if settings.MINIO_SECURE else "http"
        url = f"{scheme}://{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET}/{object_name}"
        logger.debug("Scan frame uploaded: %s", url)
        return url

    except Exception:
        logger.debug("Failed to upload scan frame")
        return None

    except Exception:
        logger.debug("Failed to upload clean frame")
        return None


def upload_vehicle_crop(
    frame: np.ndarray,
    bbox: list,
    device_id: str,
    camera_label: str,
    track_id: str,
) -> Optional[str]:
    """Crop vehicle bbox from frame and upload to MinIO."""
    client = get_minio_client()
    if client is None:
        return None

    try:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Add 20% padding around bbox for context
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * 0.2)
        pad_y = int(bh * 0.2)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None

        # Cap crop to max 30% of frame area — prevent full-frame crops for huge bboxes
        crop_area = (x2 - x1) * (y2 - y1)
        frame_area = w * h
        if crop_area > frame_area * 0.3:
            # Center crop around bbox centroid with max size
            max_side = int((frame_area * 0.3) ** 0.5)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half = max_side // 2
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half), min(h, cy + half)

        cropped = frame[y1:y2, x1:x2]
        ok, buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None

        data = buf.tobytes()
        ts = int(time.time())
        object_name = f"vehicles/{device_id}/{camera_label}/{track_id}_{ts}.jpg"

        client.put_object(
            settings.MINIO_BUCKET,
            object_name,
            io.BytesIO(data),
            len(data),
            content_type="image/jpeg",
        )

        scheme = "https" if settings.MINIO_SECURE else "http"
        return f"{scheme}://{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET}/{object_name}"

    except Exception:
        logger.exception("Failed to upload vehicle crop for %s", track_id)
        return None
