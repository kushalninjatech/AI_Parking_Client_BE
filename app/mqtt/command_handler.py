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
        # Extract action: cmd/{device_id}/snapshot → "snapshot"
        #                  cmd/{device_id}/config/camera → "config/camera"
        parts = topic.split("/")
        action = "/".join(parts[2:])  # everything after cmd/{device_id}/

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
    """Capture a frame from a camera. If slot_label is provided, crop + upload to MinIO."""
    _publish_ack(client, command_id, "snapshot", "acknowledged")

    def _do_snapshot():
        try:
            from app.db.session import SessionLocal
            from app.models.camera import Camera
            from app.models.parking_slot import ParkingSlot
            from app.camera.factory import create_camera

            import time as _time

            db = SessionLocal()
            try:
                # Find camera by label, id, or first active
                cmd_payload = payload.get("payload") or {}
                camera_label = cmd_payload.get("camera_label")
                camera_id = cmd_payload.get("camera_id")
                slot_label = cmd_payload.get("slot_label")

                if camera_label:
                    cam = db.query(Camera).filter(Camera.label == camera_label).first()
                elif camera_id:
                    cam = db.query(Camera).filter(Camera.id == int(camera_id)).first()
                else:
                    cam = db.query(Camera).filter(Camera.is_active == True).first()

                if not cam:
                    _publish_ack(client, command_id, "snapshot", "failed", error="No active camera")
                    return

                camera = create_camera(cam.source, cam.camera_type)
                if not camera.open():
                    _publish_ack(client, command_id, "snapshot", "failed", error="Failed to open camera")
                    return

                # Retry frame capture (RTSP streams may need warmup)
                ok, frame = False, None
                for attempt in range(5):
                    ok, frame = camera.read()
                    if ok and frame is not None:
                        break
                    _time.sleep(0.5)
                camera.release()

                if not ok or frame is None:
                    _publish_ack(client, command_id, "snapshot", "failed", error="Failed to capture frame after retries")
                    return

                # Slot-level snapshot: crop polygon + upload to MinIO
                if slot_label:
                    slot = db.query(ParkingSlot).filter(
                        ParkingSlot.camera_id == cam.id,
                        ParkingSlot.label == slot_label,
                    ).first()
                    if not slot or not slot.polygon_coords:
                        _publish_ack(client, command_id, "snapshot", "failed", error=f"Slot '{slot_label}' not found or no polygon")
                        return

                    from app.services.minio_service import upload_slot_image
                    polygon = slot.polygon_coords
                    if isinstance(polygon, str):
                        polygon = json.loads(polygon)

                    image_url = upload_slot_image(frame, polygon, settings.DEVICE_ID, cam.label, slot_label)
                    if not image_url:
                        _publish_ack(client, command_id, "snapshot", "failed", error="MinIO upload failed")
                        return

                    import time
                    result_topic = f"parking/{settings.DEVICE_ID}/cmd/result"
                    result_payload = {
                        "device_id": settings.DEVICE_ID,
                        "command_id": command_id,
                        "action": "slot_snapshot",
                        "camera_label": cam.label,
                        "slot_label": slot_label,
                        "image_url": image_url,
                        "timestamp": time.time(),
                    }
                    client.publish(result_topic, json.dumps(result_payload), qos=1)
                    _publish_ack(client, command_id, "snapshot", "completed")
                    return

                # Full-frame snapshot: save locally + send base64
                os.makedirs("data/snapshots", exist_ok=True)
                path = f"data/snapshots/cmd_snapshot_{cam.id}.jpg"
                cv2.imwrite(path, frame)

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
            subprocess.Popen(["sudo", "systemctl", "restart", "ai-parking-be"])
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


def _handle_config_camera(client, command_id: str, payload: dict):
    """Create/update/delete camera from Central command."""
    _publish_ack(client, command_id, "config/camera", "acknowledged")

    def _do():
        try:
            from app.db.session import SessionLocal
            from app.models.camera import Camera
            from app.core.constants import CameraType, CameraStatus

            action = payload.get("action", "create")
            camera_data = payload.get("payload", {})
            label = camera_data.get("label", "")
            source = camera_data.get("source", "0")
            camera_type = camera_data.get("camera_type", "USB")

            db = SessionLocal()
            try:
                existing = db.query(Camera).filter(Camera.label == label).first()

                if action == "delete":
                    if existing:
                        existing.is_active = False
                        db.commit()
                        # Remove from detection loop
                        from app.main import detection_loop
                        detection_loop.remove_camera(existing.id)
                    _publish_ack(client, command_id, "config/camera", "completed")
                    return

                if existing:
                    if source:
                        existing.source = source
                    existing.camera_type = CameraType(camera_type)
                    existing.is_active = True
                    existing.status = CameraStatus.ACTIVE
                    db.commit()
                    logger.info("Camera '%s' updated from Central", label)
                else:
                    cam = Camera(
                        label=label,
                        source=source,
                        camera_type=CameraType(camera_type),
                        status=CameraStatus.ACTIVE,
                    )
                    db.add(cam)
                    db.commit()
                    logger.info("Camera '%s' created from Central", label)

                _publish_ack(client, command_id, "config/camera", "completed")
            finally:
                db.close()
        except Exception as e:
            logger.exception("config/camera command failed")
            _publish_ack(client, command_id, "config/camera", "failed", error=str(e))

    threading.Thread(target=_do, daemon=True).start()


def _handle_config_slots(client, command_id: str, payload: dict):
    """Create/update slots with polygon coords from Central command."""
    _publish_ack(client, command_id, "config/slots", "acknowledged")

    def _do():
        try:
            from app.db.session import SessionLocal
            from app.models.camera import Camera
            from app.models.parking_slot import ParkingSlot
            from app.core.constants import SlotState

            slots_payload = payload.get("payload", {})
            camera_label = slots_payload.get("camera_label", "")
            slots_data = slots_payload.get("slots", [])

            db = SessionLocal()
            try:
                cam = db.query(Camera).filter(Camera.label == camera_label).first()
                if not cam:
                    _publish_ack(client, command_id, "config/slots", "failed", error=f"Camera '{camera_label}' not found")
                    return

                incoming_labels = {s["label"] for s in slots_data}

                for slot_data in slots_data:
                    label = slot_data["label"]
                    polygon_coords = slot_data.get("polygon_coords")

                    # Auto-calculate bounding box
                    pos_x1, pos_y1, pos_x2, pos_y2 = None, None, None, None
                    if polygon_coords:
                        try:
                            points = json.loads(polygon_coords) if isinstance(polygon_coords, str) else polygon_coords
                            xs = [p[0] for p in points]
                            ys = [p[1] for p in points]
                            pos_x1, pos_y1 = min(xs), min(ys)
                            pos_x2, pos_y2 = max(xs), max(ys)
                            if isinstance(polygon_coords, list):
                                polygon_coords = json.dumps(polygon_coords)
                        except (json.JSONDecodeError, IndexError):
                            pass

                    slot_type = slot_data.get("slot_type", "GENERAL")

                    existing = db.query(ParkingSlot).filter(
                        ParkingSlot.camera_id == cam.id, ParkingSlot.label == label
                    ).first()

                    if existing:
                        existing.polygon_coords = polygon_coords
                        existing.slot_type = slot_type
                        existing.pos_x1 = pos_x1
                        existing.pos_y1 = pos_y1
                        existing.pos_x2 = pos_x2
                        existing.pos_y2 = pos_y2
                    else:
                        db.add(ParkingSlot(
                            label=label,
                            camera_id=cam.id,
                            state=SlotState.EMPTY,
                            slot_type=slot_type,
                            polygon_coords=polygon_coords,
                            pos_x1=pos_x1, pos_y1=pos_y1,
                            pos_x2=pos_x2, pos_y2=pos_y2,
                        ))

                # Remove slots not in incoming list
                existing_slots = db.query(ParkingSlot).filter(ParkingSlot.camera_id == cam.id).all()
                for slot in existing_slots:
                    if slot.label not in incoming_labels:
                        db.delete(slot)

                db.commit()

                # Update detection loop
                from app.main import detection_loop
                all_slots = db.query(ParkingSlot).filter(ParkingSlot.camera_id == cam.id).all()
                slot_dicts = [
                    {"id": s.id, "label": s.label, "polygon_coords": s.polygon_coords,
                     "slot_type": s.slot_type or "GENERAL",
                     "capacity_car": s.capacity_car or 0,
                     "capacity_two_wheeler": s.capacity_two_wheeler or 0}
                    for s in all_slots
                ]
                detection_loop.add_camera(cam.id, cam.source, slot_dicts, cam.camera_type, cam.label)

                logger.info("Synced %d slots for camera '%s' from Central", len(slots_data), camera_label)
                _publish_ack(client, command_id, "config/slots", "completed")
            finally:
                db.close()
        except Exception as e:
            logger.exception("config/slots command failed")
            _publish_ack(client, command_id, "config/slots", "failed", error=str(e))

    threading.Thread(target=_do, daemon=True).start()


def _handle_calibrate(client, command_id: str, payload: dict):
    """Run calibration on a slot from Central command."""
    _publish_ack(client, command_id, "calibrate", "acknowledged")

    def _do():
        try:
            from app.db.session import SessionLocal
            from app.models.camera import Camera
            from app.models.parking_slot import ParkingSlot
            from app.models.calibration import Calibration
            from app.camera.factory import create_camera
            from app.main import parking_detector

            cal_payload = payload.get("payload", {})
            camera_label = cal_payload.get("camera_label", "")
            slot_label = cal_payload.get("slot_label", "")

            db = SessionLocal()
            try:
                cam = db.query(Camera).filter(Camera.label == camera_label).first()
                if not cam:
                    _publish_ack(client, command_id, "calibrate", "failed", error=f"Camera '{camera_label}' not found")
                    return

                slot = db.query(ParkingSlot).filter(
                    ParkingSlot.camera_id == cam.id, ParkingSlot.label == slot_label
                ).first()
                if not slot or not slot.polygon_coords:
                    _publish_ack(client, command_id, "calibrate", "failed", error=f"Slot '{slot_label}' not found or no polygon")
                    return

                # Capture frame
                camera = create_camera(cam.source, cam.camera_type)
                if not camera.open():
                    _publish_ack(client, command_id, "calibrate", "failed", error="Failed to open camera")
                    return
                ok, frame = camera.read()
                camera.release()
                if not ok or frame is None:
                    _publish_ack(client, command_id, "calibrate", "failed", error="Failed to capture frame")
                    return

                polygon = json.loads(slot.polygon_coords)
                cal_data = parking_detector.calibrate_slot(frame, slot.id, polygon)
                if not cal_data:
                    _publish_ack(client, command_id, "calibrate", "failed", error="Calibration failed")
                    return

                existing = db.query(Calibration).filter(Calibration.slot_id == slot.id).first()
                if existing:
                    existing.calibration_data = cal_data
                else:
                    db.add(Calibration(slot_id=slot.id, calibration_data=cal_data))
                db.commit()

                parking_detector.set_calibration(slot.id, cal_data)
                logger.info("Calibrated slot '%s' on camera '%s' from Central", slot_label, camera_label)
                _publish_ack(client, command_id, "calibrate", "completed")
            finally:
                db.close()
        except Exception as e:
            logger.exception("Calibrate command failed")
            _publish_ack(client, command_id, "calibrate", "failed", error=str(e))

    threading.Thread(target=_do, daemon=True).start()


def _run_script(script_name: str, args: list = None) -> tuple:
    """Run a script from the scripts/ directory. Returns (success, result_dict)."""
    import subprocess
    import pathlib

    script_path = pathlib.Path(settings.GIT_REPO_DIR) / "scripts" / script_name
    cmd = [str(script_path)] + (args or [])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        error = result.stderr.strip()
        try:
            err_data = json.loads(error)
            error = err_data.get("error", error)
        except (json.JSONDecodeError, AttributeError):
            pass
        return False, {"error": error or f"{script_name} failed"}

    try:
        return True, json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return True, {"output": result.stdout.strip()}


def _handle_update(client, command_id: str, payload: dict):
    """Run scripts/update.sh → publish result → restart service."""
    _publish_ack(client, command_id, "update", "acknowledged")

    def _do():
        import subprocess
        import time

        try:
            cmd_payload = payload.get("payload") or {}
            branch = cmd_payload.get("branch", settings.GIT_BRANCH)
            commit = cmd_payload.get("commit", "")

            args = [branch]
            if commit:
                args.append(commit)

            ok, result_data = _run_script("update.sh", args)

            if not ok:
                _publish_ack(client, command_id, "update", "failed", error=result_data.get("error"))
                return

            result_topic = f"parking/{settings.DEVICE_ID}/cmd/result"
            result_data.update({
                "device_id": settings.DEVICE_ID,
                "command_id": command_id,
                "action": "update",
                "timestamp": time.time(),
            })
            client.publish(result_topic, json.dumps(result_data), qos=1)
            _publish_ack(client, command_id, "update", "completed")

            # Restart service after ACK is sent
            time.sleep(1)
            subprocess.Popen(["sudo", "systemctl", "restart", "ai-parking-be"])

        except Exception as e:
            logger.exception("Update command failed")
            _publish_ack(client, command_id, "update", "failed", error=str(e))

    threading.Thread(target=_do, daemon=True).start()


def _handle_rollback(client, command_id: str, payload: dict):
    """Run scripts/rollback.sh → publish result → restart service."""
    _publish_ack(client, command_id, "rollback", "acknowledged")

    def _do():
        import subprocess
        import time

        try:
            cmd_payload = payload.get("payload") or {}
            target_commit = cmd_payload.get("commit", "")

            args = [target_commit] if target_commit else []

            ok, result_data = _run_script("rollback.sh", args)

            if not ok:
                _publish_ack(client, command_id, "rollback", "failed", error=result_data.get("error"))
                return

            result_topic = f"parking/{settings.DEVICE_ID}/cmd/result"
            result_data.update({
                "device_id": settings.DEVICE_ID,
                "command_id": command_id,
                "action": "rollback",
                "timestamp": time.time(),
            })
            client.publish(result_topic, json.dumps(result_data), qos=1)
            _publish_ack(client, command_id, "rollback", "completed")

            # Restart service after ACK is sent
            time.sleep(1)
            subprocess.Popen(["sudo", "systemctl", "restart", "ai-parking-be"])

        except Exception as e:
            logger.exception("Rollback command failed")
            _publish_ack(client, command_id, "rollback", "failed", error=str(e))

    threading.Thread(target=_do, daemon=True).start()


def _handle_version(client, command_id: str, payload: dict):
    """Run scripts/version.sh → publish result."""
    _publish_ack(client, command_id, "version", "acknowledged")

    def _do():
        import time

        try:
            ok, result_data = _run_script("version.sh")

            if not ok:
                _publish_ack(client, command_id, "version", "failed", error=result_data.get("error"))
                return

            result_topic = f"parking/{settings.DEVICE_ID}/cmd/result"
            result_data.update({
                "device_id": settings.DEVICE_ID,
                "command_id": command_id,
                "action": "version",
                "timestamp": time.time(),
            })
            client.publish(result_topic, json.dumps(result_data), qos=1)
            _publish_ack(client, command_id, "version", "completed")

        except Exception as e:
            logger.exception("Version command failed")
            _publish_ack(client, command_id, "version", "failed", error=str(e))

    threading.Thread(target=_do, daemon=True).start()


_HANDLERS = {
    "snapshot": _handle_snapshot,
    "restart": _handle_restart,
    "update": _handle_update,
    "rollback": _handle_rollback,
    "version": _handle_version,
    "config": _handle_config,
    "config/camera": _handle_config_camera,
    "config/slots": _handle_config_slots,
    "calibrate": _handle_calibrate,
}
