"""Environment settings used by Reachy diagnostics."""

import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class KitSettings:
    """Small subset of legacy ambient-home settings needed by hardware checks."""

    daemon_status_url: str = "http://127.0.0.1:8000/api/daemon/status"
    events_dir: Path = Path("~/Library/Logs/ambient-home/events").expanduser()
    wake_word: str = "hey_jarvis"
    wake_threshold: float = 0.5
    hub_command: str | None = None

    @classmethod
    def from_env(cls) -> "KitSettings":
        """Read compatible environment variables and a cwd .env file."""
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        threshold_raw = os.getenv("AMBIENT_WAKE_THRESHOLD", "0.5")
        try:
            threshold = float(threshold_raw)
        except ValueError as exc:
            raise ValueError("AMBIENT_WAKE_THRESHOLD must be a number") from exc
        return cls(
            daemon_status_url=os.getenv("AMBIENT_DAEMON_STATUS_URL", "http://127.0.0.1:8000/api/daemon/status"),
            events_dir=Path(os.getenv("AMBIENT_EVENTS_DIR") or "~/Library/Logs/ambient-home/events").expanduser(),
            wake_word=os.getenv("AMBIENT_WAKE_WORD", "hey_jarvis"),
            wake_threshold=threshold,
            hub_command=os.getenv("AMBIENT_HUB_COMMAND") or None,
        )
