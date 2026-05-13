"""Sync cameras, slots, and detection states to the central platform."""

import json
import logging
from typing import Dict, List, Optional

import urllib.request
import urllib.error

from app.core.config import settings

logger = logging.getLogger(__name__)


class CentralSync:

    def _request(self, method: str, url: str, payload: Dict) -> Optional[Dict]:
        if not settings.CENTRAL_API_URL or not settings.CENTRAL_API_TOKEN:
            logger.debug("Central API not configured, skipping")
            return None
        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.CENTRAL_API_TOKEN}",
                },
                method=method,
            )
            resp = urllib.request.urlopen(req, timeout=10)
            return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            logger.error("Central API %s %s → HTTP %d: %s", method, url, e.code, e.read().decode()[:200])
        except Exception:
            logger.exception("Central API request failed: %s %s", method, url)
        return None

    def register_camera(self, camera_data: Dict) -> Optional[str]:
        """Register camera on central. Returns central_camera_id or None."""
        url = f"{settings.CENTRAL_API_URL}/devices/{settings.DEVICE_ID}/cameras"
        payload = {
            "label": camera_data["label"],
            "source": camera_data["source"],
            "camera_type": camera_data["camera_type"],
            "lot_id": settings.LOT_ID,
        }
        resp = self._request("POST", url, payload)
        if resp:
            central_id = resp.get("id") or resp.get("camera_id")
            if central_id:
                logger.info("Camera '%s' registered on central: %s", camera_data["label"], central_id)
                return str(central_id)
        return None

    def register_slot(self, central_camera_id: str, slot_data: Dict) -> Optional[str]:
        """Register slot on central. Returns central_slot_id or None."""
        url = f"{settings.CENTRAL_API_URL}/cameras/{central_camera_id}/slots"
        payload = {
            "label": slot_data["label"],
            "polygon_coords": slot_data.get("polygon_coords"),
            "pos_x1": slot_data.get("pos_x1"),
            "pos_y1": slot_data.get("pos_y1"),
            "pos_x2": slot_data.get("pos_x2"),
            "pos_y2": slot_data.get("pos_y2"),
        }
        resp = self._request("POST", url, payload)
        if resp:
            central_id = resp.get("id") or resp.get("slot_id")
            if central_id:
                logger.info("Slot '%s' registered on central: %s", slot_data["label"], central_id)
                return str(central_id)
        return None

    def push_slot_config(self, central_camera_id: str, slots: List[Dict]) -> bool:
        """Push updated polygon coords for existing central slots."""
        url = f"{settings.CENTRAL_API_URL}/cameras/{central_camera_id}/slot-config"
        payload = {
            "slots": [
                {
                    "slot_id": s["central_slot_id"],
                    "polygon_coords": s.get("polygon_coords"),
                    "pos_x1": s.get("pos_x1"),
                    "pos_y1": s.get("pos_y1"),
                    "pos_x2": s.get("pos_x2"),
                    "pos_y2": s.get("pos_y2"),
                }
                for s in slots if s.get("central_slot_id")
            ],
        }
        if not payload["slots"]:
            return False
        return self._request("POST", url, payload) is not None

    def push_slot_states(self, central_camera_id: str, results: List[Dict]) -> bool:
        """Push current slot detection states to central after each cycle."""
        url = f"{settings.CENTRAL_API_URL}/cameras/{central_camera_id}/slot-states"
        payload = {
            "device_id": settings.DEVICE_ID,
            "lot_id": settings.LOT_ID,
            "slots": [
                {
                    "slot_id": r["central_slot_id"],
                    "state": r["state"].value if hasattr(r["state"], "value") else r["state"],
                    "confidence": round(float(r.get("confidence", 0.0)), 4),
                }
                for r in results if r.get("central_slot_id")
            ],
        }
        if not payload["slots"]:
            logger.debug("No central slot IDs mapped — skipping state push")
            return False
        ok = self._request("POST", url, payload) is not None
        if ok:
            logger.info("Pushed %d slot states to central camera %s", len(payload["slots"]), central_camera_id)
        return ok


central_sync = CentralSync()
