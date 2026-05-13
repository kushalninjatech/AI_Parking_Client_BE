"""System telemetry collection for heartbeat payloads."""

import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)

_BOOT_TIME: float = time.time()


def collect_telemetry() -> Dict:
    """Collect CPU, temperature, memory, and disk metrics."""
    data: Dict = {}

    try:
        import psutil

        data["cpu_percent"] = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        data["memory_percent"] = mem.percent
        disk = psutil.disk_usage("/")
        data["disk_percent"] = disk.percent
        data["uptime_seconds"] = round(time.time() - psutil.boot_time())
    except ImportError:
        data["uptime_seconds"] = round(time.time() - _BOOT_TIME)
    except Exception:
        logger.debug("Failed to collect psutil metrics", exc_info=True)

    # RPi temperature — /sys/class/thermal (Linux only)
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            data["temperature"] = round(int(f.read().strip()) / 1000.0, 1)
    except (FileNotFoundError, ValueError, OSError):
        pass

    # macOS temperature fallback via psutil sensors
    if "temperature" not in data:
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            if temps:
                for entries in temps.values():
                    if entries:
                        data["temperature"] = entries[0].current
                        break
        except (ImportError, AttributeError):
            pass

    return data
