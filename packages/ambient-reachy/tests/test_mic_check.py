"""Microphone diagnostic tests."""

import sys
import threading
import subprocess

import numpy as np
import pytest

from ambient_reachy import mic_check
from ambient_reachy.mic_check import MicProbe, MicProbeTimeout, verdict, measure_frames


def test_measure_frames_downmixes_stereo() -> None:
    frame = np.column_stack(
        (
            np.full(1600, 0.5, dtype=np.float32),
            np.full(1600, 0.25, dtype=np.float32),
        )
    )

    rms, peak = measure_frames([frame])

    assert rms == np.float32(0.375)
    assert peak == np.float32(0.375)


def test_measure_frames_empty() -> None:
    assert measure_frames([]) == (0.0, 0.0)


def test_verdict() -> None:
    assert verdict(0.0, 0.0) == "silent"
    assert verdict(1e-4, 9e-4) == "floor-only"
    assert verdict(0.1, 0.1) == "ok"


def _probe(frames: int, peak: float) -> MicProbe:
    return MicProbe(frames=frames, samples=frames * 160, rms=peak / 2, peak=peak, verdict=verdict(0.0, peak))


def test_warm_gstreamer_returns_elapsed_time_and_runs_current_python(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    monotonic = iter([10.0, 12.5])

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(mic_check.time, "monotonic", lambda: next(monotonic))

    elapsed = mic_check.warm_gstreamer(timeout_s=30.0, run=run)

    assert elapsed == 2.5
    command, kwargs = calls[0]
    assert command[:2] == [sys.executable, "-c"]
    assert "Gst.init([])" in command[2]
    assert "ElementFactory.find('appsink')" in command[2]
    assert kwargs == {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "timeout": 30.0,
        "check": False,
    }


def test_warm_gstreamer_raises_on_timeout() -> None:
    def timeout(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        raise subprocess.TimeoutExpired(command, 15.0)

    with pytest.raises(mic_check.GStreamerWarmupError, match="not ready within 15 s"):
        mic_check.warm_gstreamer(timeout_s=15.0, run=timeout)


def test_warm_gstreamer_raises_on_nonzero_exit() -> None:
    def failed(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        return subprocess.CompletedProcess(command, 3, b"", b"")

    with pytest.raises(mic_check.GStreamerWarmupError, match="exited with code 3"):
        mic_check.warm_gstreamer(run=failed)


def test_probe_microphone_bounded_returns_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _probe(100, 0.2)
    monkeypatch.setattr(mic_check, "probe_microphone", lambda seconds: probe)

    assert mic_check.probe_microphone_bounded(1.0, timeout_s=1.0) is probe


def test_probe_microphone_bounded_times_out_and_releases_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocked_probe(seconds: float) -> MicProbe:
        del seconds
        started.set()
        release.wait()
        return _probe(100, 0.2)

    monkeypatch.setattr(mic_check, "probe_microphone", blocked_probe)
    try:
        with pytest.raises(MicProbeTimeout, match="microphone gave no result within 0 s"):
            mic_check.probe_microphone_bounded(1.0, timeout_s=0.05)
        assert started.is_set()
    finally:
        release.set()


def test_probe_microphone_bounded_propagates_probe_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_probe(seconds: float) -> MicProbe:
        del seconds
        raise ValueError("probe failed")

    monkeypatch.setattr(mic_check, "probe_microphone", fail_probe)

    with pytest.raises(ValueError, match="probe failed"):
        mic_check.probe_microphone_bounded(1.0, timeout_s=1.0)


def test_ensure_microphone_audio_skips_reboot_when_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(mic_check, "probe_microphone", lambda seconds: _probe(100, 0.2))
    monkeypatch.setattr(mic_check, "restart_audio_processor", lambda **kwargs: pytest.fail("must not reboot"))

    assert (
        mic_check.ensure_microphone_audio(
            emit=lambda name, component, level="info", **data: events.append((name, data))
        )
        == "ok"
    )
    assert events == [("mic.check", {"verdict": "ok", "frames": 100, "peak": 0.2, "stage": "initial"})]


def test_ensure_microphone_audio_reboots_once_when_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    probes = iter([_probe(100, 0.0), _probe(100, 0.2)])
    restarts: list[bool] = []
    monkeypatch.setattr(mic_check, "probe_microphone", lambda seconds: next(probes))
    monkeypatch.setattr(mic_check, "restart_audio_processor", lambda **kwargs: restarts.append(True) or True)

    assert (
        mic_check.ensure_microphone_audio(
            emit=lambda name, component, level="info", **data: events.append((name, data))
        )
        == "ok"
    )
    assert restarts == [True]
    assert [event[0] for event in events] == ["mic.check", "mic.check"]
    assert [event[1]["stage"] for event in events] == ["initial", "after_restart"]


def test_ensure_microphone_audio_reports_persistent_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mic_check, "probe_microphone", lambda seconds: _probe(100, 0.0))
    monkeypatch.setattr(mic_check, "restart_audio_processor", lambda **kwargs: True)

    assert mic_check.ensure_microphone_audio() == "silent"


def test_ensure_microphone_audio_no_frames_does_not_reboot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mic_check, "probe_microphone", lambda seconds: _probe(0, 0.0))
    monkeypatch.setattr(mic_check, "restart_audio_processor", lambda **kwargs: pytest.fail("must not reboot"))

    assert mic_check.ensure_microphone_audio() == "silent"


def test_ensure_microphone_audio_recovers_from_initial_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    restarts: list[bool] = []
    probes = iter([MicProbeTimeout("timeout"), _probe(100, 0.2)])

    def probe_bounded(seconds: float, timeout_s: float | None = None) -> MicProbe:
        del seconds, timeout_s
        result = next(probes)
        if isinstance(result, MicProbeTimeout):
            raise result
        return result

    monkeypatch.setattr(mic_check, "probe_microphone_bounded", probe_bounded)
    monkeypatch.setattr(mic_check, "restart_audio_processor", lambda **kwargs: restarts.append(True) or True)

    assert (
        mic_check.ensure_microphone_audio(
            emit=lambda name, component, level="info", **data: events.append((name, data))
        )
        == "ok"
    )
    assert restarts == [True]
    assert events == [
        ("mic.check", {"verdict": "timeout", "frames": 0, "peak": 0.0, "stage": "initial"}),
        ("mic.check", {"verdict": "ok", "frames": 100, "peak": 0.2, "stage": "after_restart"}),
    ]


def test_ensure_microphone_audio_times_out_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, str, dict[str, object]]] = []
    monkeypatch.setattr(
        mic_check,
        "probe_microphone_bounded",
        lambda seconds, timeout_s=None: (_ for _ in ()).throw(MicProbeTimeout("no result")),
    )
    monkeypatch.setattr(mic_check, "restart_audio_processor", lambda **kwargs: True)

    result = mic_check.ensure_microphone_audio(
        emit=lambda name, component, level="info", **data: events.append((name, level, data))
    )

    assert result == "timeout"
    assert events == [
        ("mic.check", "warning", {"verdict": "timeout", "frames": 0, "peak": 0.0, "stage": "initial"}),
        ("mic.check", "error", {"verdict": "timeout", "frames": 0, "peak": 0.0, "stage": "after_restart"}),
    ]


def test_ensure_microphone_audio_timeout_without_restart_device_returns_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mic_check,
        "probe_microphone_bounded",
        lambda seconds, timeout_s=None: (_ for _ in ()).throw(MicProbeTimeout("no result")),
    )
    monkeypatch.setattr(mic_check, "restart_audio_processor", lambda **kwargs: False)

    assert mic_check.ensure_microphone_audio() == "timeout"


def test_format_firmware_version() -> None:
    assert mic_check.format_firmware_version([0, 2, 1, 4]) == "2.1.4"
    assert mic_check.format_firmware_version((2, 1, 2)) == "2.1.2"
    assert mic_check.format_firmware_version("bad reply") == "bad reply"
