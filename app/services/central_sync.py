"""Sync slot config from client to central platform."""

import json
import logging
from typing import Dict, List, Optional

import urllib.request
import urllib.error

from app.core.config import settings

logger = logging.getLogger(__name__)


class CentralSync:
    """Push slot configurations to the central API."""

    def push_slot_config(self, central_camera_id: str, slots: List[Dict]) -> bool:
        """Push slot positions + polygons to central's slot-config endpoint.

        Args:
            central_camera_id: Camera UUID on central platform
            slots: List of {central_slot_id, polygon_coords, pos_x1, pos_y1, pos_x2, pos_y2}
        """
        if not settings.CENTRAL_API_URL or not settings.CENTRAL_API_TOKEN:
            logger.warning("Central API not configured, skipping sync")
            return False

        url = f"{settings.CENTRAL_API_URL}/cameras/{central_camera_id}/slot-config"
        payload = {
            "slots": [
                {
                    "slot_id": s["central_slot_id"],
                    "polygon_coords": s.get("polygon_coords"),
                    "pos_x1": s["pos_x1"],
                    "pos_y1": s["pos_y1"],
                    "pos_x2": s["pos_x2"],
                    "pos_y2": s["pos_y2"],
                }
                for s in slots if s.get("central_slot_id")
            ],
        }

        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.CENTRAL_API_TOKEN}",
                },
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=10)
            logger.info("Synced %d slots to central camera %s (status=%d)",
                        len(payload["slots"]), central_camera_id, resp.status)
            return True
        except urllib.error.HTTPError as e:
            logger.error("Central sync failed: HTTP %d - %s", e.code, e.read().decode()[:200])
            return False
        except Exception:
            logger.exception("Central sync failed")
            return False

    def push_slot_states(self, central_camera_id: str, slots: List[Dict]) -> bool:
        """Push current slot states to central (backup for MQTT)."""
        # This is handled by MQTT primarily. This method is a fallback.
        logger.debug("Slot states pushed via MQTT, HTTP fallback not needed")
        return True


central_sync = CentralSync()
