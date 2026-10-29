"""Hardware test runner tests (no robot needed)."""

import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pytest
from reachy_mini.utils import create_head_pose

from ambient_reachy import hw_test
from ambient_reachy._events import EventLog
from ambient_reachy.hw_test import (
    FAIL,
    PASS,
    SKIP,
    Check,
    PausedApp,
    CheckFailed,
    CheckResult,
    run_checks,
    wake_phrase,
    check_telemetry,
    format_markdown,
    rotation_between_deg,
)


def _fail() -> str:
    raise CheckFailed("broken")


def _crash() -> str:
    raise OSError("no device")


def test_run_checks_skips_dependents_of_failures() -> None:
    results = run_checks(
        [
            Check("a", lambda: "fine"),
            Check("b", _fail),
            Check("c", lambda: "fine", needs=("b",)),
            Check("d", _crash, needs=("a",)),
        ]
    )

    assert [(r.name, r.status) for r in results] == [("a", PASS), ("b", FAIL), ("c", SKIP), ("d", FAIL)]
    assert results[1].detail == "broken"
    assert results[2].detail == "needs b"
    assert results[3].detail == "OSError: no device"


def test_run_checks_scrubs_details() -> None:
    results = run_checks([Check("a", lambda: "token sk-abcdefghijklmnop in /Users/nithin/x")])

    assert "sk-" not in results[0].detail
    assert "nithin" not in results[0].detail


def test_wake_phrase() -> None:
    assert wake_phrase("hey_alfred") == "hey alfred"
    assert wake_phrase("/models/hey_jarvis.onnx") == "hey jarvis"


def test_rotation_between_deg_ignores_starting_tilt() -> None:
    start = create_head_pose(roll=15, yaw=-20, pitch=-25, degrees=True)

    assert rotation_between_deg(start, start @ create_head_pose(pitch=10, degrees=True)) == pytest.approx(10.0)
    assert rotation_between_deg(np.eye(4), np.eye(4)) == pytest.approx(0.0, abs=1e-6)


def test_check_telemetry_round_trip(tmp_path: Path) -> None:
    log = EventLog(tmp_path)

    assert "read back" in check_telemetry(log, "run-1")
    assert [event.type for event in log.read(datetime(2000, 1, 1, tzinfo=timezone.utc))] == ["hw_test.probe"]


class FakeLaunchctl:
    def __init__(self, loaded: bool, bootstrap_rc: int = 0) -> None:
        self.loaded = loaded
        self.bootstrap_rc = bootstrap_rc
        self.calls: list[str] = []

    def __call__(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        self.calls.append(args[0])
        rc = {"print": 0 if self.loaded else 113, "bootout": 0, "bootstrap": self.bootstrap_rc}[args[0]]
        return subprocess.CompletedProcess(["launchctl", *args], rc, "", "")


def test_paused_app_restarts_even_when_checks_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hw_test.time, "sleep", lambda seconds: None)
    fake = FakeLaunchctl(loaded=True)
    pause = PausedApp(run=fake, system=lambda: "Darwin")

    with pytest.raises(ValueError), pause:
        raise ValueError("boom")

    assert fake.calls == ["print", "bootout", "bootstrap"]
    assert pause.restored is True


def test_paused_app_leaves_unloaded_agent_alone() -> None:
    fake = FakeLaunchctl(loaded=False)

    with PausedApp(run=fake, system=lambda: "Darwin") as pause:
        pass

    assert fake.calls == ["print"]
    assert pause.paused is False
    assert pause.restored is None


def test_paused_app_noop_off_macos_or_disabled() -> None:
    fake = FakeLaunchctl(loaded=True)

    with PausedApp(run=fake, system=lambda: "Linux"):
        pass
    with PausedApp(enabled=False, run=fake, system=lambda: "Darwin"):
        pass

    assert fake.calls == []


def test_format_markdown_reports_failure_and_restart() -> None:
    text = format_markdown(
        [CheckResult("mic", PASS, "ok", 1.0), CheckResult("head", FAIL, "a|b", 2.0)],
        total_s=42.0,
        app_restored=True,
    )

    assert "### `ambient-reachy-hw-test`: FAIL (42 s)" in text
    assert "| head | FAIL | a\\|b |" in text
    assert "restarted afterwards: yes" in text


def test_main_writes_reports_and_returns_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMBIENT_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.setattr(
        hw_test,
        "build_checks",
        lambda *args, **kwargs: [Check("ok", lambda: "fine"), Check("bad", _fail)],
    )

    code = hw_test.main(["--no-pause", "--json", str(tmp_path / "r.json"), "--markdown", str(tmp_path / "r.md")])

    assert code == 1
    payload = json.loads((tmp_path / "r.json").read_text())
    assert payload["ok"] is False
    assert [c["status"] for c in payload["checks"]] == [PASS, FAIL]
    assert "FAIL" in (tmp_path / "r.md").read_text()
    events = EventLog(tmp_path / "events").read(datetime(2000, 1, 1, tzinfo=timezone.utc))
    assert [e.type for e in events] == ["hw_test.check", "hw_test.check", "hw_test.completed"]
    assert events[-1].data["ok"] is False


def test_wake_check_skips_without_hub_command(tmp_path: Path) -> None:
    from ambient_reachy.hw_test import CheckSkipped, check_wake_word
    from ambient_reachy.settings import KitSettings

    with pytest.raises(CheckSkipped, match="needs --hub-command \\(wake word runs in ambient-home\\)"):
        check_wake_word(KitSettings(events_dir=tmp_path), EventLog(tmp_path), 1.0, False)


def test_live_check_requires_hub_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ambient_reachy.hw_test import CheckFailed, check_live
    from ambient_reachy.settings import KitSettings

    with pytest.raises(CheckFailed, match="--live requires AMBIENT_HUB_COMMAND"):
        check_live(KitSettings(events_dir=tmp_path), EventLog(tmp_path), 1.0, False)
    monkeypatch.delenv("AMBIENT_HUB_COMMAND", raising=False)
    assert hw_test.main(["--live", "--no-pause"]) == 2


def test_live_check_waits_for_live_session_and_preserves_paid_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from ambient_reachy.settings import KitSettings

    event_log = EventLog(tmp_path / "events")
    sleeps: list[float] = []

    class Process:
        polls = 0
        terminated = False
        env: dict[str, str] = {}

        def poll(self) -> int | None:
            self.polls += 1
            if self.polls == 1:
                event_log.emit("app.started", "app")
            elif self.polls == 2:
                event_log.emit("live.session_started", "live")
            elif self.polls == 3:
                event_log.emit("tool.completed", "live", tool="dispatch_job")
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float) -> int:
            del timeout
            return 0

        def kill(self) -> None:
            self.terminated = True

    process = Process()

    def popen(command, *, env, **kwargs):
        del kwargs
        assert command == ["python", "-m", "ambient_home", "--profile", "a profile"]
        process.env = env
        return process

    monkeypatch.setenv("OPENAI_API_KEY", "fake-paid-test-key")
    monkeypatch.setenv("DEVIN_API_KEY", "fake-devin-test-key")
    monkeypatch.setenv("DEVIN_ORG_ID", "fake-org-test-id")
    monkeypatch.setenv("AMBIENT_GARAGE_URL", "https://fake-garage.invalid")
    monkeypatch.setenv("AMBIENT_DATA_DIR", "/existing/ambient-data")
    monkeypatch.setenv("AMBIENT_DEVICES_FILE", "/existing/devices.toml")
    monkeypatch.setattr(hw_test.time, "sleep", sleeps.append)
    detail = hw_test.check_live(
        KitSettings(
            events_dir=event_log.directory,
            wake_word="hey_jarvis",
            hub_command='python -m ambient_home --profile "a profile"',
        ),
        event_log,
        15.0,
        False,
        popen=popen,
        hub_env=(("AMBIENT_IDLE_TIMEOUT_S", "11"), ("REACHY_TEST_ENV", "live=value")),
    )

    assert detail == "Live session opened; WARNING 1 dispatch_job call(s) during the test"
    assert sleeps[0] == 0.5
    assert 3.0 in sleeps
    assert "say 'hey jarvis' now" in capsys.readouterr().out
    assert process.env["OPENAI_API_KEY"] == "fake-paid-test-key"
    assert process.env["AMBIENT_EVENTS_DIR"] == str(event_log.directory)
    assert process.env["AMBIENT_IDLE_TIMEOUT_S"] == "11"
    assert process.env["AMBIENT_MAX_SESSION_S"] == "20"
    assert process.env["AMBIENT_HEARTBEAT_S"] == "30"
    assert process.env["DEVIN_API_KEY"] == "fake-devin-test-key"
    assert process.env["DEVIN_ORG_ID"] == "fake-org-test-id"
    assert process.env["AMBIENT_GARAGE_URL"] == "https://fake-garage.invalid"
    assert process.env["AMBIENT_DATA_DIR"] == "/existing/ambient-data"
    assert process.env["AMBIENT_DEVICES_FILE"] == "/existing/devices.toml"
    assert process.env["REACHY_TEST_ENV"] == "live=value"
    assert process.terminated


def test_live_check_fails_on_early_exit_and_session_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ambient_reachy.hw_test import CheckFailed, check_live
    from ambient_reachy.settings import KitSettings

    settings = KitSettings(events_dir=tmp_path / "events", hub_command="fake-hub")
    event_log = EventLog(settings.events_dir)

    class Process:
        returncode: int | None = 7
        terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float) -> int:
            del timeout
            return 0

        def kill(self) -> None:
            self.terminated = True

    process = Process()
    with pytest.raises(CheckFailed, match="hub exited early with code 7"):
        check_live(settings, event_log, 1.0, False, popen=lambda *args, **kwargs: process)

    process.returncode = None
    process.terminated = False
    clock = [0.0]

    def emit_started_on_poll(*args, **kwargs):
        del args, kwargs

        class StartedProcess(Process):
            returncode = None
            polls = 0

            def poll(self) -> int | None:
                self.polls += 1
                if self.polls == 1:
                    event_log.emit("app.started", "app")
                return None

        return StartedProcess()

    monkeypatch.setattr(hw_test.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(hw_test.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    with pytest.raises(CheckFailed, match="no Live session opened before the timeout"):
        check_live(settings, event_log, 4.0, False, popen=emit_started_on_poll)


def test_speech_heard_prompts_or_uses_say(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from ambient_reachy import mic_check
    from ambient_reachy.hw_test import check_speech_heard

    monkeypatch.setattr(mic_check, "probe_microphone", lambda seconds: mic_check.MicProbe(10, 1600, 0.1, 0.2, "ok"))
    monkeypatch.setattr(hw_test.shutil, "which", lambda name: "/usr/bin/say")
    monkeypatch.setattr(
        hw_test.subprocess,
        "Popen",
        lambda command, **kwargs: type(
            "Speaker",
            (),
            {"poll": lambda self: 0, "terminate": lambda self: None},
        )(),
    )

    assert "verdict=ok" in check_speech_heard(0.1, "hey_jarvis", False)
    assert "say 'hey jarvis' now" in capsys.readouterr().out
    assert "verdict=ok" in check_speech_heard(0.1, "hey_jarvis", True)


def test_hub_wake_blanks_provider_keys_and_preserves_unrelated_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ambient_reachy.settings import KitSettings

    event_log = EventLog(tmp_path / "events")

    class Process:
        returncode = None
        polls = 0
        env: dict[str, str] = {}
        terminated = False

        def poll(self) -> int | None:
            self.polls += 1
            if self.polls == 1:
                event_log.emit("app.started", "app")
            elif self.polls == 2:
                event_log.emit("wake.detected", "live")
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float) -> int:
            del timeout
            return 0

        def kill(self) -> None:
            self.terminated = True

    process = Process()

    def popen(command, *, env, **kwargs):
        del kwargs
        assert command == ["python", "-m", "ambient_home"]
        process.env = env
        return process

    monkeypatch.setattr(hw_test.shutil, "which", lambda name: None)
    monkeypatch.setenv("OPENAI_API_KEY", "paid-key-must-not-leak")
    monkeypatch.setenv("DEVIN_API_KEY", "devin-key-must-not-leak")
    monkeypatch.setenv("DEVIN_ORG_ID", "org-id-must-not-leak")
    monkeypatch.setenv("AMBIENT_GARAGE_URL", "https://garage.invalid")
    monkeypatch.setenv("AMBIENT_DATA_DIR", "/existing/ambient-data")
    monkeypatch.setenv("AMBIENT_DEVICES_FILE", "/existing/devices.toml")
    detail = hw_test.check_wake_word(
        KitSettings(events_dir=tmp_path / "events", hub_command="python -m ambient_home"),
        event_log,
        1.0,
        False,
        popen=popen,
        hub_env=(("REACHY_TEST_ENV", "free=value"),),
    )

    assert "wake detected by hub" in detail
    assert process.env["OPENAI_API_KEY"] == ""
    assert process.env["DEVIN_API_KEY"] == ""
    assert process.env["DEVIN_ORG_ID"] == ""
    assert process.env["AMBIENT_GARAGE_URL"] == ""
    assert process.env["AMBIENT_EVENTS_DIR"] == str(tmp_path / "events")
    assert process.env["AMBIENT_DATA_DIR"] == "/existing/ambient-data"
    assert process.env["AMBIENT_DEVICES_FILE"] == "/existing/devices.toml"
    assert process.env["REACHY_TEST_ENV"] == "free=value"
    assert process.terminated


@pytest.mark.parametrize("key", ["OPENAI_API_KEY", "DEVIN_API_KEY", "DEVIN_ORG_ID", "AMBIENT_GARAGE_URL"])
def test_hub_env_cannot_reset_free_check_secrets(key: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        hw_test.main(["--hub-env", f"{key}=must-not-be-set"])

    assert exit_info.value.code == 2
    assert f"--hub-env cannot override {key} during the free wake check" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("value", "message"),
    [("NO_EQUALS", "expected KEY=VALUE"), ("=empty-key", "KEY must not be empty")],
)
def test_hub_env_malformed_values_exit_two(value: str, message: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        hw_test.main(["--hub-env", value])

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_build_checks_passes_hub_env_to_both_hub_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ambient_reachy.settings import KitSettings

    received: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    pairs = (("AMBIENT_DATA_DIR", "/isolated/data"), ("AMBIENT_DEVICES_FILE", "/isolated/devices.toml"))

    def free_check(*args, hub_env=(), **kwargs) -> str:
        del args, kwargs
        received.append(("wake", tuple(hub_env)))
        return "wake"

    def live_check(*args, hub_env=(), **kwargs) -> str:
        del args, kwargs
        received.append(("live", tuple(hub_env)))
        return "live"

    monkeypatch.setattr(hw_test, "check_daemon", lambda settings: "daemon")
    monkeypatch.setattr(hw_test, "check_audio_device", lambda: "audio")
    monkeypatch.setattr(hw_test, "check_microphone", lambda seconds: "mic")
    monkeypatch.setattr(hw_test, "check_speech_heard", lambda *args: "speech")
    monkeypatch.setattr(hw_test, "check_wake_word", free_check)
    monkeypatch.setattr(hw_test, "check_head", lambda: "head")
    monkeypatch.setattr(hw_test, "check_live", live_check)
    checks = hw_test.build_checks(
        KitSettings(events_dir=tmp_path / "events", hub_command="fake-hub"),
        EventLog(tmp_path / "events"),
        "run",
        mic_seconds=1,
        wake_timeout_s=1,
        speak=False,
        live=True,
        live_timeout_s=1,
        hub_env=pairs,
    )

    run_checks(checks)

    assert received == [("wake", pairs), ("live", pairs)]


def test_main_parses_repeatable_hub_env_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMBIENT_EVENTS_DIR", str(tmp_path / "events"))
    captured: dict[str, object] = {}

    def build_checks(*args, **kwargs):
        del args
        captured.update(kwargs)
        return [Check("ok", lambda: "fine")]

    monkeypatch.setattr(hw_test, "build_checks", build_checks)

    assert (
        hw_test.main(["--no-pause", "--hub-env", "CUSTOM=first=value", "--hub-env", "AMBIENT_DATA_DIR=/isolated"]) == 0
    )
    assert captured["hub_env"] == [("CUSTOM", "first=value"), ("AMBIENT_DATA_DIR", "/isolated")]


def test_hub_env_help_documents_per_run_isolation(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        hw_test.main(["--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--hub-env" in help_text
    assert "AMBIENT_DATA_DIR" in help_text
    assert "AMBIENT_DEVICES_FILE" in help_text


def test_hub_wake_fails_on_timeout_or_early_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ambient_reachy.hw_test import CheckFailed
    from ambient_reachy.settings import KitSettings

    settings = KitSettings(events_dir=tmp_path / "events", hub_command="fake-hub")
    log = EventLog(tmp_path / "events")
    clock = [0.0]
    monkeypatch.setattr(hw_test.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(hw_test.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    class Process:
        returncode = None
        terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float) -> int:
            del timeout
            return 0

        def kill(self) -> None:
            self.terminated = True

    process = Process()
    with pytest.raises(CheckFailed, match="did not emit app.started"):
        hw_test.check_wake_word(settings, log, 1.0, False, popen=lambda *args, **kwargs: process)
    assert process.terminated

    process.returncode = 1
    clock[0] = 0.0
    with pytest.raises(CheckFailed, match="hub exited early with code 1"):
        hw_test.check_wake_word(settings, log, 1.0, False, popen=lambda *args, **kwargs: process)


def test_settled_pose_waits_for_pose_to_stop_moving(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hw_test.time, "sleep", lambda seconds: None)
    pitches = iter([2.0, 6.0, 9.0, 9.9, 10.0, 10.0])

    pose = hw_test.settled_pose(lambda: create_head_pose(pitch=next(pitches), degrees=True))

    assert rotation_between_deg(np.eye(4), pose) == pytest.approx(10.0)


def _mic_probe(verdict: str, frames: int = 100) -> object:
    from ambient_reachy.mic_check import MicProbe

    return MicProbe(
        frames=frames, samples=frames * 160, rms=0.0, peak=0.01 if verdict == "ok" else 0.0, verdict=verdict
    )


def test_check_microphone_reboots_xmos_once_when_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    from ambient_reachy import mic_check

    probes = iter([_mic_probe("silent"), _mic_probe("ok")])
    reboots: list[bool] = []
    monkeypatch.setattr(mic_check, "probe_microphone", lambda seconds: next(probes))
    monkeypatch.setattr(mic_check, "restart_audio_processor", lambda **kwargs: reboots.append(True) or True)

    assert "silent until XMOS reboot" in hw_test.check_microphone(1.0)
    assert reboots == [True]


def test_check_microphone_fails_when_still_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    from ambient_reachy import mic_check

    monkeypatch.setattr(mic_check, "probe_microphone", lambda seconds: _mic_probe("silent"))
    monkeypatch.setattr(mic_check, "restart_audio_processor", lambda **kwargs: True)

    with pytest.raises(CheckFailed, match="cable"):
        hw_test.check_microphone(1.0)
