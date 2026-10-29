"""Validated, hardware-independent Reachy connector configuration."""

from math import isfinite
from dataclasses import dataclass
from collections.abc import Mapping


DEFAULT_BODY_TOOLS = (
    "dance",
    "stop_dance",
    "play_emotion",
    "stop_emotion",
    "move_head",
    "sweep_look",
    "head_tracking",
)
_ALLOWED_KEYS = {
    "daemon_status_url",
    "daemon_wait_s",
    "daemon_poll_s",
    "daemon_watchdog_s",
    "daemon_watchdog_failures",
    "mic_autorecover",
    "mic_check_seconds",
    "silence_alarm_s",
    "camera",
    "body_tools",
    "motion_duration_s",
}


@dataclass(frozen=True)
class ReachyConfig:
    """Settings for one Reachy Mini Lite device."""

    daemon_status_url: str = "http://127.0.0.1:8000/api/daemon/status"
    daemon_wait_s: float = 60.0
    daemon_poll_s: float = 1.0
    daemon_watchdog_s: float = 5.0
    daemon_watchdog_failures: int = 3
    mic_autorecover: bool = True
    mic_check_seconds: float = 1.5
    silence_alarm_s: float = 5.0
    camera: bool = True
    body_tools: tuple[str, ...] = DEFAULT_BODY_TOOLS
    motion_duration_s: float = 1.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ReachyConfig":
        """Build settings, rejecting unknown keys and values with invalid types."""
        unknown = set(values) - _ALLOWED_KEYS
        if unknown:
            raise ValueError(f"unknown Reachy config keys: {sorted(unknown)}")
        defaults = cls()
        result: dict[str, object] = {}
        for name in _ALLOWED_KEYS:
            value = values.get(name, getattr(defaults, name))
            if name == "daemon_status_url":
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{name} must be a non-empty string")
            elif name in {"mic_autorecover", "camera"}:
                if not isinstance(value, bool):
                    raise ValueError(f"{name} must be a boolean")
            elif name == "daemon_watchdog_failures":
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must be an integer >= 1")
            elif name == "body_tools":
                if not isinstance(value, (list, tuple)) or any(not isinstance(tool, str) for tool in value):
                    raise ValueError("body_tools must be a list of strings")
                unknown_tools = set(value) - set(DEFAULT_BODY_TOOLS)
                if unknown_tools:
                    raise ValueError(f"unknown body tools: {sorted(unknown_tools)}")
                if len(set(value)) != len(value):
                    raise ValueError("body_tools must not contain duplicates")
                value = tuple(value)
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be a number > 0")
                value = float(value)
            result[name] = value
        return cls(**result)  # type: ignore[arg-type]
