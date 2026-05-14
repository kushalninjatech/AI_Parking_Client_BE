"""MQTT command handler — receives and executes commands from Central."""

import base64
import json
import logging
import os
import threading

import cv2

from app.core.config import settings

logger = logging.getLogger(__name__)


def handle_command(client, userdata, msg):
    """MQTT on_message callback for cmd/{DEVICE_ID}/# topics."""
    try:
        topic = msg.topic
        payload = json.loads(msg.payload.decode())
        command_id = payload.get("command_id", "")
        action = topic.rsplit("/", 1)[-1]  # "snapshot", "restart", "config", etc.

        logger.info("Command received: %s (id=%s)", action, command_id)

        handler = _HANDLERS.get(action)
        if handler:
            handler(client, command_id, payload)
        else:
            logger.warning("Unknown command action: %s", action)
            _publish_ack(client, command_id, action, "failed", error=f"Unknown action: {action}")

    except json.JSONDecodeError:
        logger.error("Invalid JSON in command payload on %s", msg.topic)
    except Exception:
        logger.exception("Error handling command on %s", msg.topic)


def _publish_ack(client, command_id: str, action: str, status: str, error: str = None):
    """Publish command ACK back to Central."""
    import time
    topic = f"parking/{settings.DEVICE_ID}/ack"
    payload = {
        "device_id": settings.DEVICE_ID,
        "command_id": command_id,
        "action": action,
        "status": status,
    }
    if error:
        payload["error"] = error
    payload["timestamp"] = time.time()
    client.publish(topic, json.dumps(payload), qos=1)
    logger.info("ACK sent: %s → %s", action, status)


def _handle_snapshot(client, command_id: str, payload: dict):
    """Capture a frame from the first active camera and return as base64."""
    _publish_ack(client, command_id, "snapshot", "acknowledged")

    def _do_snapshot():
        try:
            from app.db.session import SessionLocal
            from app.models.camera import Camera
            from app.camera.factory import create_camera

            db = SessionLocal()
            try:
                # Use camera_id from payload, or first active camera
                camera_id = (payload.get("payload") or {}).get("camera_id")
                if camera_id:
                    cam = db.query(Camera).filter(Camera.id == int(camera_id), Camera.is_active == True).first()
                else:
                    cam = db.query(Camera).filter(Camera.is_active == True).first()

                if not cam:
                    _publish_ack(client, command_id, "snapshot", "failed", error="No active camera")
                    return

                camera = create_camera(cam.source, cam.camera_type)
                if not camera.open():
                    _publish_ack(client, command_id, "snapshot", "failed", error="Failed to open camera")
                    return

                ok, frame = camera.read()
                camera.release()

                if not ok or frame is None:
                    _publish_ack(client, command_id, "snapshot", "failed", error="Failed to capture frame")
                    return

                # Save locally
                os.makedirs("data/snapshots", exist_ok=True)
                path = f"data/snapshots/cmd_snapshot_{cam.id}.jpg"
                cv2.imwrite(path, frame)

                # Encode and publish result
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(jpeg.tobytes()).decode()

                import time
                result_topic = f"parking/{settings.DEVICE_ID}/cmd/result"
                result_payload = {
                    "device_id": settings.DEVICE_ID,
                    "command_id": command_id,
                    "action": "snapshot",
                    "camera_label": cam.label,
                    "image_b64": b64,
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "timestamp": time.time(),
                }
                client.publish(result_topic, json.dumps(result_payload), qos=1)
                _publish_ack(client, command_id, "snapshot", "completed")

            finally:
                db.close()

        except Exception as e:
            logger.exception("Snapshot command failed")
            _publish_ack(client, command_id, "snapshot", "failed", error=str(e))

    # Run in thread to avoid blocking MQTT loop
    threading.Thread(target=_do_snapshot, daemon=True).start()


def _handle_restart(client, command_id: str, payload: dict):
    """Restart the client service."""
    _publish_ack(client, command_id, "restart", "acknowledged")

    def _do_restart():
        import time
        import subprocess
        time.sleep(1)  # let ACK publish
        try:
            subprocess.Popen(["sudo", "systemctl", "restart", "ai-parking-client"])
            _publish_ack(client, command_id, "restart", "completed")
        except Exception as e:
            _publish_ack(client, command_id, "restart", "failed", error=str(e))

    threading.Thread(target=_do_restart, daemon=True).start()


def _handle_config(client, command_id: str, payload: dict):
    """Update runtime config (e.g., detection_interval)."""
    _publish_ack(client, command_id, "config", "acknowledged")
    config_data = payload.get("payload", {})
    if not config_data:
        _publish_ack(client, command_id, "config", "failed", error="Empty config payload")
        return

    # TODO: apply config changes at runtime
    logger.info("Config update received: %s", config_data)
    _publish_ack(client, command_id, "config", "completed")


_HANDLERS = {
    "snapshot": _handle_snapshot,
    "restart": _handle_restart,
    "config": _handle_config,
}
