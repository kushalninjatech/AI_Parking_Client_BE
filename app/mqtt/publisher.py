"""MQTT publisher — sends slot states, heartbeats, and snapshots to central broker.

Offline resilience: when the broker is unreachable, messages are queued in the
local SQLite mqtt_outbox table and drained automatically on reconnect.
"""

import json
import logging
import threading
import time
from typing import Callable, Dict, List, Optional

import paho.mqtt.client as mqtt
from paho.mqtt.properties import Properties
from paho.mqtt.packettypes import PacketTypes

from app.core.config import settings
from app.core.constants import SlotState, SlotType, VehicleType

logger = logging.getLogger(__name__)

_RECONNECT_MIN_DELAY = 5
_RECONNECT_MAX_DELAY = 120
_DRAIN_BATCH = 100       # messages per drain cycle
_OUTBOX_MAX_AGE_DAYS = 2  # discard events older than this to avoid replay flood


class MQTTPublisher:
    def __init__(self, db_factory: Optional[Callable] = None) -> None:
        self._db_factory = db_factory
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._shutdown = False
        self._reconnecting = False
        self._reconnect_lock = threading.Lock()

    def connect(self) -> bool:
        self._shutdown = False
        try:
            self._client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"edge-{settings.DEVICE_ID}",
                protocol=mqtt.MQTTv5,
                reconnect_on_failure=False,  # we manage reconnect manually
            )
            self._client.username_pw_set(settings.MQTT_USERNAME, settings.MQTT_PASSWORD)

            # LWT — broker publishes this automatically if our connection drops
            lwt_topic = f"parking/{settings.DEVICE_ID}/status"
            self._client.will_set(lwt_topic, json.dumps({
                "device_id": settings.DEVICE_ID,
                "status": "offline",
                "timestamp": time.time(),
            }), qos=1, retain=True)

            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message

            # MQTT 5.0: persistent session (clean_start=False) — broker buffers
            # QoS 1 messages while we're offline. SessionExpiryInterval tells the
            # broker how long to keep our session (24h here).
            props = Properties(PacketTypes.CONNECT)
            props.SessionExpiryInterval = 86400

            self._client.connect(
                settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT,
                keepalive=60, clean_start=False, properties=props,
            )
            self._client.loop_start()
            logger.info("MQTT connecting to %s:%d", settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT)
            return True
        except Exception:
            logger.exception("MQTT connection failed")
            return False

    def disconnect(self) -> None:
        self._shutdown = True
        if self._client:
            # Publish clean online→offline status before disconnecting
            try:
                lwt_topic = f"parking/{settings.DEVICE_ID}/status"
                self._client.publish(lwt_topic, json.dumps({
                    "device_id": settings.DEVICE_ID,
                    "status": "offline",
                    "timestamp": time.time(),
                }), qos=1, retain=True)
            except Exception:
                pass
            self._client.loop_stop()
            self._client.disconnect()
            logger.info("MQTT disconnected")

    def publish_slot_events(self, camera_label: str, changes: List[Dict]) -> None:
        """Publish only changed slots — fires every detection cycle when state changes.

        Not retained: events are immutable history. Central should append to a log.
        """
        if not changes:
            return
        topic = f"parking/{settings.DEVICE_ID}/events"
        payload = {
            "device_id": settings.DEVICE_ID,
            "camera_id": camera_label,
            "changes": [
                {
                    "slot_label": s["label"],
                    "state": s["state"].value if isinstance(s["state"], SlotState) else s["state"],
                    "confidence": round(float(s.get("confidence", 0.0)), 4),
                    "slot_type": s["slot_type"].value if isinstance(s.get("slot_type"), SlotType) else s.get("slot_type"),
                    "detected_vehicle_type": s["detected_vehicle_type"].value if isinstance(s.get("detected_vehicle_type"), VehicleType) else s.get("detected_vehicle_type"),
                    "is_mismatched": s.get("is_mismatched", False),
                    "image_url": s.get("image_url"),
                    "occupied_car": s.get("occupied_car", 0),
                    "occupied_two_wheeler": s.get("occupied_two_wheeler", 0),
                }
                for s in changes
            ],
            "timestamp": time.time(),
        }
        self._publish(topic, payload)

    def publish_slot_snapshot(self, camera_label: str, slots: List[Dict]) -> None:
        """Publish full slot state for a camera — runs on a timer (every MQTT_SNAPSHOT_INTERVAL).

        Retained: any reconnecting consumer immediately gets the latest known state.
        Also serves as a liveness signal — no snapshot for >2 intervals = device down.
        """
        topic = f"parking/{settings.DEVICE_ID}/slots"
        payload = {
            "device_id": settings.DEVICE_ID,
            "camera_id": camera_label,
            "slots": [
                {
                    "slot_label": s["label"],
                    "state": s["state"].value if isinstance(s["state"], SlotState) else s["state"],
                    "slot_type": s.get("slot_type", "GENERAL"),
                    "detected_vehicle_type": s.get("detected_vehicle_type"),
                    "occupied_car": s.get("occupied_car", 0),
                    "occupied_two_wheeler": s.get("occupied_two_wheeler", 0),
                }
                for s in slots
            ],
            "timestamp": time.time(),
        }
        # Retained snapshot: broker keeps last copy per topic for new subscribers.
        # If offline, drop silently — the next snapshot tick (within MQTT_SNAPSHOT_INTERVAL)
        # will reconcile. Critical changes go via publish_slot_events which IS queued.
        if self._client and self._connected:
            self._client.publish(topic, json.dumps(payload), qos=1, retain=True)
        else:
            logger.debug("Snapshot skipped (offline) — next tick will reconcile")

    def publish_status(self) -> None:
        """Publish online status on /status topic (retained). Also used by LWT for offline."""
        topic = f"parking/{settings.DEVICE_ID}/status"
        payload = {
            "device_id": settings.DEVICE_ID,
            "status": "online",
            "timestamp": time.time(),
        }
        if self._client and self._connected:
            self._client.publish(topic, json.dumps(payload), qos=1, retain=True)
        else:
            self._store_outbox(topic, payload)

    def publish_heartbeat(self, telemetry: Dict) -> None:
        """Publish system telemetry on /heartbeat topic (not retained)."""
        topic = f"parking/{settings.DEVICE_ID}/heartbeat"
        payload = {
            "device_id": settings.DEVICE_ID,
            **telemetry,
            "timestamp": time.time(),
        }
        self._publish(topic, payload)

    def publish_vehicle_events(self, camera_label: str, events: List[Dict]) -> None:
        """Publish vehicle entry/exit events for multi-capacity zones."""
        if not events:
            return
        topic = f"parking/{settings.DEVICE_ID}/vehicle_events"
        payload = {
            "device_id": settings.DEVICE_ID,
            "camera_id": camera_label,
            "events": [
                {
                    "event_type": e["event_type"],
                    "slot_label": e["slot_label"],
                    "vehicle_type": e["vehicle_type"],
                    "track_id": e["track_id"],
                    "confidence": round(float(e.get("confidence", 0.0)), 4),
                    "image_url": e.get("image_url"),
                    "duration_seconds": e.get("duration_seconds"),
                }
                for e in events
            ],
            "timestamp": time.time(),
        }
        self._publish(topic, payload)

    # ------------------------------------------------------------------
    # Config sync (camera + slot CRUD → Central)
    # ------------------------------------------------------------------

    def publish_sync_camera(self, action: str, camera_data: Dict) -> None:
        """Sync camera create/update/delete to Central."""
        topic = f"parking/{settings.DEVICE_ID}/sync/camera"
        payload = {
            "device_id": settings.DEVICE_ID,
            "action": action,
            "camera": camera_data,
            "timestamp": time.time(),
        }
        self._publish(topic, payload)

    def publish_sync_slots(self, action: str, camera_label: str, slots_data: List[Dict]) -> None:
        """Sync slots + polygon_coords to Central."""
        topic = f"parking/{settings.DEVICE_ID}/sync/slots"
        payload = {
            "device_id": settings.DEVICE_ID,
            "action": action,
            "camera_label": camera_label,
            "slots": slots_data,
            "timestamp": time.time(),
        }
        self._publish(topic, payload)

    # ------------------------------------------------------------------
    # Command ACK
    # ------------------------------------------------------------------

    def publish_ack(self, command_id: str, action: str, status: str) -> None:
        topic = f"parking/{settings.DEVICE_ID}/ack"
        payload = {
            "device_id": settings.DEVICE_ID,
            "command_id": command_id,
            "action": action,
            "status": status,
        }
        self._publish(topic, payload)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _publish(self, topic: str, payload: Dict) -> None:
        if self._client and self._connected:
            self._client.publish(topic, json.dumps(payload), qos=1)
            logger.debug("Published to %s", topic)
        else:
            self._store_outbox(topic, payload)

    def _store_outbox(self, topic: str, payload: Dict) -> None:
        if not self._db_factory:
            logger.warning("MQTT offline, no outbox — dropped: %s", topic)
            return
        db = self._db_factory()
        try:
            from app.models.mqtt_outbox import MqttOutbox
            db.add(MqttOutbox(topic=topic, payload=json.dumps(payload)))
            db.commit()
            logger.info("MQTT offline → queued in outbox: %s", topic)
        except Exception:
            logger.exception("Failed to write MQTT outbox")
            db.rollback()
        finally:
            db.close()

    def _drain_outbox(self) -> None:
        if not self._db_factory:
            return
        db = self._db_factory()
        try:
            from app.models.mqtt_outbox import MqttOutbox
            from datetime import datetime, timedelta

            # Discard stale messages to avoid replaying ancient history
            cutoff = datetime.utcnow() - timedelta(days=_OUTBOX_MAX_AGE_DAYS)
            stale = db.query(MqttOutbox).filter(MqttOutbox.created_at < cutoff).count()
            if stale:
                db.query(MqttOutbox).filter(MqttOutbox.created_at < cutoff).delete()
                db.commit()
                logger.warning("Discarded %d stale outbox messages (>%dd old)", stale, _OUTBOX_MAX_AGE_DAYS)

            pending = db.query(MqttOutbox).order_by(MqttOutbox.id).limit(_DRAIN_BATCH).all()
            if not pending:
                return

            logger.info("Draining %d queued MQTT messages…", len(pending))
            sent = 0
            for msg in pending:
                if not self._connected:
                    logger.warning("Lost connection mid-drain — will resume on next reconnect")
                    break
                self._client.publish(msg.topic, msg.payload, qos=1)
                db.delete(msg)
                sent += 1

            db.commit()
            logger.info("Drained %d/%d outbox messages", sent, len(pending))
        except Exception:
            logger.exception("MQTT outbox drain failed")
            db.rollback()
        finally:
            db.close()

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code == 0:
            self._connected = True
            self._reconnecting = False
            logger.info("MQTT connected")

            # Publish online status (retained so central sees it immediately)
            status_topic = f"parking/{settings.DEVICE_ID}/status"
            client.publish(status_topic, json.dumps({
                "device_id": settings.DEVICE_ID,
                "status": "online",
                "timestamp": time.time(),
            }), qos=1, retain=True)

            # Subscribe to commands
            cmd_topic = f"cmd/{settings.DEVICE_ID}/#"
            client.subscribe(cmd_topic, qos=1)
            logger.info("Subscribed to %s", cmd_topic)

            # Drain queued messages in background (don't block the MQTT loop thread)
            threading.Thread(target=self._drain_outbox, daemon=True).start()
        else:
            logger.error("MQTT connect failed: %s", reason_code)

    def _on_disconnect(self, _client, _userdata, _flags, reason_code, _properties) -> None:
        self._connected = False
        rc = reason_code.value if hasattr(reason_code, "value") else int(reason_code)

        if rc == 142:
            # Session taken over by a newer instance with the same client_id.
            # Do NOT reconnect — the other instance is the intended one.
            logger.warning(
                "MQTT session taken over (rc=142) by a newer instance. "
                "This instance will not reconnect."
            )
            return

        logger.warning("MQTT disconnected: rc=%d (%s) — will reconnect with backoff", rc, reason_code)
        if not self._shutdown:
            with self._reconnect_lock:
                if self._reconnecting:
                    logger.debug("Reconnect loop already running — skipping")
                    return
                self._reconnecting = True
            threading.Thread(target=self._reconnect_loop, daemon=True).start()

    def _on_message(self, client, userdata, msg) -> None:
        """Route incoming MQTT messages (commands from Central)."""
        if msg.topic.startswith("cmd/"):
            from app.mqtt.command_handler import handle_command
            handle_command(client, userdata, msg)

    def _reconnect_loop(self) -> None:
        delay = _RECONNECT_MIN_DELAY
        try:
            while not self._shutdown and not self._connected:
                logger.info("MQTT reconnecting in %ds…", delay)
                time.sleep(delay)
                if self._shutdown or self._connected:
                    break
                try:
                    # Full teardown — kill old client completely
                    try:
                        self._client.loop_stop()
                        self._client.disconnect()
                    except Exception:
                        pass

                    # Build a brand-new client (same as connect())
                    self._client = mqtt.Client(
                        mqtt.CallbackAPIVersion.VERSION2,
                        client_id=f"edge-{settings.DEVICE_ID}",
                        protocol=mqtt.MQTTv5,
                        reconnect_on_failure=False,
                    )
                    self._client.username_pw_set(settings.MQTT_USERNAME, settings.MQTT_PASSWORD)

                    lwt_topic = f"parking/{settings.DEVICE_ID}/status"
                    self._client.will_set(lwt_topic, json.dumps({
                        "device_id": settings.DEVICE_ID,
                        "status": "offline",
                        "timestamp": time.time(),
                    }), qos=1, retain=True)

                    self._client.on_connect = self._on_connect
                    self._client.on_disconnect = self._on_disconnect
                    self._client.on_message = self._on_message

                    props = Properties(PacketTypes.CONNECT)
                    props.SessionExpiryInterval = 86400

                    self._client.connect(
                        settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT,
                        keepalive=60, clean_start=False, properties=props,
                    )
                    self._client.loop_start()

                    # Wait up to 10s for _on_connect to fire
                    for _ in range(20):
                        if self._connected:
                            break
                        time.sleep(0.5)

                    if self._connected:
                        logger.info("MQTT reconnected successfully (fresh client)")
                        delay = _RECONNECT_MIN_DELAY
                    else:
                        logger.warning("MQTT reconnect: no CONNACK within 10s")
                        delay = min(delay * 2, _RECONNECT_MAX_DELAY)
                except Exception as exc:
                    logger.warning("MQTT reconnect failed: %s", exc)
                    delay = min(delay * 2, _RECONNECT_MAX_DELAY)
        finally:
            with self._reconnect_lock:
                self._reconnecting = False
