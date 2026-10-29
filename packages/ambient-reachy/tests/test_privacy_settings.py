"""Telemetry privacy, compatibility event shape, and settings tests."""

import json
from pathlib import Path
from datetime import datetime, timezone

from ambient_reachy._scrub import scrub_text
from ambient_reachy._events import EventLog
from ambient_reachy.settings import KitSettings


def test_event_log_preserves_raw_event_fields_and_day_file(tmp_path: Path) -> None:
    timestamp = datetime(2025, 3, 4, 12, 30, tzinfo=timezone.utc)
    log = EventLog(tmp_path / "events", clock=lambda: timestamp)

    event = log.emit("hw_test.probe", "hw_test", run_id="run-1", ok=True)
    path = tmp_path / "events" / "2025-03-04.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert event is not None
    assert set(payload) == {"ts", "type", "component", "level", "data"}
    assert payload["type"] == "hw_test.probe"
    assert payload["component"] == "hw_test"
    assert payload["level"] == "info"
    assert payload["data"] == {"run_id": "run-1", "ok": True}
    assert log.read(timestamp)[0] == event


def test_emit_scrubs_sensitive_values_and_drops_content_fields(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events")
    sensitive = {
        "email": "alice@example.com",
        "phone": "+1 (415) 555-0199",
        "openai_key": "sk-liveAbCdEf1234567890",
        "devin_key": "apk_user_AbCdEf1234567890",
        "authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.tokenvalue",
        "blob": "a1b2c3d4e5f6a7b8c9d0" * 2,
        "home_path": str(Path.home() / "private" / "file.txt"),
        "other_path": "/Users/alice/private/file.txt",
        "public_ip": "203.0.113.42",
        "transcript": "private spoken words",
        "request": "private requested task",
    }

    assert log.emit("test.privacy", "test", **sensitive) is not None
    raw = next((tmp_path / "events").glob("*.jsonl")).read_bytes()

    for value in (
        "alice@example.com",
        "415) 555-0199",
        "sk-liveAbCdEf1234567890",
        "apk_user_AbCdEf1234567890",
        "eyJhbGciOiJIUzI1NiJ9.tokenvalue",
        "a1b2c3d4e5f6a7b8c9d0",
        str(Path.home()),
        "/Users/alice",
        "203.0.113.42",
        "private spoken words",
        "private requested task",
    ):
        assert value.encode() not in raw
    assert b'"transcript"' not in raw
    assert b'"request"' not in raw


def test_scrub_patterns_and_harmless_values() -> None:
    cases = {
        "mail alice@example.com": "<email>",
        "call +1 415-555-0199": "<phone>",
        "key sk-AbCdEf1234567890": "<secret>",
        "key apk_user_AbCdEf1234567890": "<secret>",
        "key ghp_AbCdEf1234567890": "<secret>",
        "key gho_AbCdEf1234567890": "<secret>",
        "key github_pat_AbCdEf1234567890": "<secret>",
        "header Bearer eyJhbGciOiJIUzI1NiJ9": "<secret>",
        "Authorization: Basic abc123": "<secret>",
        "mixed a1b2c3d4e5f6a7b8c9d0a1b2c3d4e5f6": "<secret>",
        "ip 203.0.113.42": "<ip>",
        "/Users/alice/project": "/Users/<user>/project",
        "/home/bob/project": "/home/<user>/project",
    }
    for value, replacement in cases.items():
        assert replacement in scrub_text(value)

    assert scrub_text("verdict ok") == "verdict ok"
    assert scrub_text("127.0.0.1") == "127.0.0.1"
    assert scrub_text("x" * 501) == "x" * 500 + "…"
    assert scrub_text(str(Path.home() / "secret.txt")) == "~/secret.txt"


def test_settings_reads_env_and_dotenv_without_overriding(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("AMBIENT_WAKE_WORD=hey_from_file\nAMBIENT_EVENTS_DIR=~/events\n")
    monkeypatch.setenv("AMBIENT_WAKE_WORD", "hey_from_env")
    monkeypatch.setenv("AMBIENT_DAEMON_STATUS_URL", "http://127.0.0.1:9000/status")
    monkeypatch.setenv("AMBIENT_WAKE_THRESHOLD", "0.7")
    monkeypatch.setenv("AMBIENT_HUB_COMMAND", "python -m ambient_home")
    monkeypatch.delenv("AMBIENT_EVENTS_DIR", raising=False)

    settings = KitSettings.from_env()

    assert settings.wake_word == "hey_from_env"
    assert settings.daemon_status_url.endswith(":9000/status")
    assert settings.wake_threshold == 0.7
    assert settings.events_dir == Path("~/events").expanduser()
    assert settings.hub_command == "python -m ambient_home"
