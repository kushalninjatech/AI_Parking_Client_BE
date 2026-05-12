"""MQTT publisher — sends slot states, heartbeats, and snapshots to central broker."""

import json
import logging
import time
from typing import Dict, List, Optional

import paho.mqtt.client as mqtt

from app.core.config import settings
from app.core.constants import SlotState

logger = logging.getLogger(__name__)


class MQTTPublisher:
    def __init__(self) -> None:
        self._client: Optional[mqtt.Client] = None
        self._connected = False

    def connect(self) -> bool:
        try:
            self._client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"edge-{settings.DEVICE_ID}",
                protocol=mqtt.MQTTv5,
            )
            self._client.username_pw_set(settings.MQTT_USERNAME, settings.MQTT_PASSWORD)
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.connect(settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT, keepalive=60)
            self._client.loop_start()
            logger.info("MQTT connecting to %s:%d", settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT)
            return True
        except Exception:
            logger.exception("MQTT connection failed")
            return False

    def disconnect(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            logger.info("MQTT disconnected")

    def publish_slot_states(self, camera_label: str, slots: List[Dict]) -> None:
        """Publish all slot states for a camera."""
        topic = f"parking/{settings.LOT_ID}/{settings.DEVICE_ID}/slots"
        payload = {
            "device_id": settings.DEVICE_ID,
            "camera_id": camera_label,
            "slots": [
                {"slot_label": s["label"], "state": s["state"].value if isinstance(s["state"], SlotState) else s["state"]}
                for s in slots
            ],
            "timestamp": time.time(),
        }
        self._publish(topic, payload)

    def publish_heartbeat(self) -> None:
        """Publish device heartbeat."""
        topic = f"parking/{settings.LOT_ID}/{settings.DEVICE_ID}/heartbeat"
        payload = {
            "device_id": settings.DEVICE_ID,
            "timestamp": time.time(),
        }
        self._publish(topic, payload)

    def publish_ack(self, command_id: str, action: str, status: str) -> None:
        """Publish command acknowledgement."""
        topic = f"parking/{settings.LOT_ID}/{settings.DEVICE_ID}/ack"
        payload = {
            "device_id": settings.DEVICE_ID,
            "command_id": command_id,
            "action": action,
            "status": status,
        }
        self._publish(topic, payload)

    def _publish(self, topic: str, payload: Dict) -> None:
        if self._client and self._connected:
            self._client.publish(topic, json.dumps(payload), qos=1)
            logger.debug("Published to %s", topic)
        else:
            logger.warning("MQTT not connected, message dropped: %s", topic)

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code == 0:
            self._connected = True
            logger.info("MQTT connected")
            # Subscribe to commands
            cmd_topic = f"cmd/{settings.DEVICE_ID}/#"
            client.subscribe(cmd_topic, qos=1)
            logger.info("Subscribed to %s", cmd_topic)
        else:
            logger.error("MQTT connect failed: %s", reason_code)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        self._connected = False
        logger.warning("MQTT disconnected: %s", reason_code)
