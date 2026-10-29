"""Minimal JSONL event storage compatible with ambient-home's EventLog format."""

import json
import logging
from pathlib import Path
from datetime import date, datetime, timezone
from dataclasses import field, dataclass
from collections.abc import Callable

from ambient_reachy._scrub import scrub_data


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    """One timestamped event."""

    ts: datetime
    type: str
    component: str
    level: str = "info"
    data: dict[str, str | int | float | bool | None] = field(default_factory=dict)


class EventLog:
    """Thread-safe, day-partitioned event writer and reader."""

    def __init__(
        self,
        directory: Path,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.directory = directory
        self.clock = clock

    def emit(
        self,
        type: str,
        component: str,
        level: str = "info",
        **data: object,
    ) -> Event | None:
        """Append one event, returning None when disk I/O fails."""
        try:
            timestamp = self.clock().astimezone(timezone.utc)
            event = Event(timestamp, type, component, level, scrub_data(data))
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{timestamp:%Y-%m-%d}.jsonl"
            payload = {
                "ts": event.ts.isoformat(),
                "type": event.type,
                "component": event.component,
                "level": event.level,
                "data": event.data,
            }
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
            return event
        except (OSError, ValueError, TypeError):
            logger.debug("Could not write hardware-check event", exc_info=True)
            return None

    def read(self, since: datetime, until: datetime | None = None) -> list[Event]:
        """Read well-formed events between inclusive timestamps."""
        if since.tzinfo is None or since.utcoffset() is None:
            raise ValueError("since must be timezone-aware")
        start = since.astimezone(timezone.utc)
        end_value = until or self.clock()
        if end_value.tzinfo is None or end_value.utcoffset() is None:
            raise ValueError("until must be timezone-aware")
        end = end_value.astimezone(timezone.utc)
        if start > end:
            return []
        events: list[Event] = []
        for ordinal in range(start.date().toordinal(), end.date().toordinal() + 1):
            path = self.directory / f"{date.fromordinal(ordinal):%Y-%m-%d}.jsonl"
            try:
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        try:
                            raw = json.loads(line)
                            timestamp = datetime.fromisoformat(raw["ts"])
                            if timestamp.tzinfo is None:
                                continue
                            event = Event(
                                timestamp.astimezone(timezone.utc),
                                raw["type"],
                                raw["component"],
                                raw["level"],
                                raw["data"],
                            )
                        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                            logger.debug("Skipping malformed event in %s", path)
                            continue
                        if start <= event.ts <= end:
                            events.append(event)
            except FileNotFoundError:
                continue
            except OSError:
                logger.debug("Could not read hardware-check events from %s", path, exc_info=True)
        return sorted(events, key=lambda event: event.ts)
