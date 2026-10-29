"""Reachy Device protocol tests using only fake hardware."""

import sys
import asyncio
import logging
import subprocess
from typing import Any
from pathlib import Path
from collections import deque
from collections.abc import Iterable

import numpy as np
import pytest
from ambient_device_sdk import (
    Capability,
    DeviceConfig,
    HealthStatus,
    SessionState,
    DeviceContext,
    DeviceCreateError,
    load_plugin,
    create_device,
    available_kinds,
)
from ambient_device_sdk.testing import run_device_contract, check_device_contract

from ambient_reachy.device import ReachyDevice
from ambient_reachy.body_tools import build_body_tools


class FakeMedia:
    def __init__(self, frames: Iterable[tuple[int, np.ndarray]] = (), *, fail_play_start: bool = False) -> None:
        self.frames = deque(frames)
        self.sample_rate = self.frames[0][0] if self.frames else 16_000
        self.fail_play_start = fail_play_start
        self.pushed: list[np.ndarray] = []
        self.clears = 0
        self.recording = False
        self.playing = False
        self.audio = self

    def start_recording(self) -> None:
        self.recording = True

    def stop_recording(self) -> None:
        self.recording = False

    def start_playing(self) -> None:
        if self.fail_play_start:
            raise RuntimeError("failed to start playback")
        self.playing = True

    def stop_playing(self) -> None:
        self.playing = False

    def get_audio_sample(self) -> np.ndarray | None:
        if not self.frames:
            return None
        self.sample_rate, frame = self.frames.popleft()
        return frame

    def get_input_audio_samplerate(self) -> int:
        return self.sample_rate

    def push_audio_sample(self, samples: np.ndarray) -> None:
        self.pushed.append(samples.copy())

    def clear_player(self) -> None:
        self.clears += 1

    def get_frame_jpeg(self) -> bytes:
        return b"fake-jpeg"


class FakeRobot:
    def __init__(self, media: FakeMedia, calls: list[str]) -> None:
        self.media = media
        self.calls = calls
        self.closed = False

    def goto_sleep(self) -> None:
        self.calls.append("goto_sleep")

    def enable_motors(self) -> None:
        self.calls.append("enable_motors")

    def wake_up(self) -> None:
        self.calls.append("wake_up")

    def disable_wobbling(self) -> None:
        self.calls.append("disable_wobbling")

    def __exit__(self, *args: object) -> None:
        self.closed = True
        self.calls.append("close")


class FakeMovementManager:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def start(self) -> None:
        self.calls.append("manager.start")

    def stop(self, *, reset_to_neutral: bool) -> None:
        self.calls.append(f"manager.stop:{reset_to_neutral}")

    def set_listening(self, value: bool) -> None:
        self.calls.append(f"listening:{value}")

    def set_speaking(self, value: bool) -> None:
        self.calls.append(f"speaking:{value}")

    def set_head_tracking(self, value: bool) -> None:
        self.calls.append(f"head_tracking:{value}")


class FakeHardware:
    def __init__(self, robots: Iterable[FakeRobot]) -> None:
        self.robots = deque(robots)
        self.calls: list[str] = []
        self.mic_checks: list[float] = []

    def make_robot(self) -> FakeRobot:
        return self.robots.popleft()

    def make_movement_manager(self, robot: FakeRobot) -> FakeMovementManager:
        del robot
        return FakeMovementManager(self.calls)

    def check_microphone(self, seconds: float, emit: Any) -> str:
        del emit
        self.mic_checks.append(seconds)
        return "ok"


def _robot(frames: Iterable[tuple[int, np.ndarray]] = (), *, fail_play_start: bool = False) -> FakeRobot:
    return FakeRobot(FakeMedia(frames, fail_play_start=fail_play_start), [])


def _patch_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    from ambient_reachy import daemon

    async def ready(*args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return True

    async def watch(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        await asyncio.Event().wait()

    monkeypatch.setattr(daemon, "wait_for_daemon", ready)
    monkeypatch.setattr(daemon, "watch_daemon", watch)


def _device(
    robot: FakeRobot,
    *,
    emit: Any = None,
    config: dict[str, object] | None = None,
) -> ReachyDevice:
    hardware = FakeHardware([robot])
    context = DeviceContext(
        "reachy-test",
        {"mic_autorecover": False, "camera": False, "body_tools": [], **(config or {})},
        emit=emit or (lambda *args, **kwargs: None),
    )
    return ReachyDevice(context, hardware=hardware)


async def test_sdk_contract_static_and_runtime_with_fake_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_daemon(monkeypatch)
    robot = _robot([(16_000, np.full(320, 0.25, dtype=np.float32))])
    hardware = FakeHardware([robot])
    from ambient_reachy.plugin import PLUGIN

    device = PLUGIN.create(
        DeviceContext(
            "reachy-test",
            {"mic_autorecover": False, "camera": False, "body_tools": []},
        )
    )
    assert not hardware.calls
    assert len(hardware.robots) == 1
    device._hardware = hardware

    assert check_device_contract(device) == []
    assert device.capabilities == frozenset({Capability.LISTEN, Capability.SPEAK, Capability.MOVE})
    assert await run_device_contract(device) == []
    assert robot.closed


async def test_start_stop_restart_are_idempotent_and_use_fresh_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_daemon(monkeypatch)
    first = _robot()
    second = _robot()
    hardware = FakeHardware([first, second])
    device = ReachyDevice(DeviceContext("reachy-test", {"mic_autorecover": False, "camera": False}), hardware=hardware)

    await device.stop()
    await device.start()
    first_queue = device._mic_queue
    await device.start()
    first_task = device._mic_task
    await device.stop()
    await device.stop()
    assert first.closed
    assert first_task is not None and first_task.done()

    await device.start()
    assert device._mic_queue is not first_queue
    await device.stop()
    assert second.closed
    assert not any(
        task.get_name().startswith("reachy-test-")
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
    )


async def test_camera_capability_and_look_tool_are_dependency_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_daemon(monkeypatch)
    robot = _robot()
    device = ReachyDevice(
        DeviceContext("reachy-test", {"mic_autorecover": False, "camera": True, "body_tools": []}),
        hardware=FakeHardware([robot]),
    )
    assert Capability.SEE in device.capabilities
    assert check_device_contract(device) == []
    await device.start()

    assert device.tools() == []
    await device.stop()


async def test_failed_startup_releases_robot_and_allows_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_daemon(monkeypatch)
    failed = _robot(fail_play_start=True)
    retry = _robot()
    hardware = FakeHardware([failed, retry])
    device = ReachyDevice(DeviceContext("reachy-test", {"mic_autorecover": False}), hardware=hardware)

    with pytest.raises(RuntimeError, match="failed to start playback"):
        await device.start()
    assert failed.closed
    assert device._mic_task is None

    await device.start()
    assert await device.health()
    await device.stop()
    assert retry.closed


async def test_audio_worker_emits_single_silence_alarm_and_recovers_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_daemon(monkeypatch)
    events: list[tuple[str, str, dict[str, object]]] = []

    def emit(name: str, component: str, level: str = "info", **data: object) -> None:
        events.append((name, level, data))

    samples = np.zeros(80_000, dtype=np.float32)
    robot = _robot([(16_000, samples), (16_000, samples)])
    device = _device(robot, emit=emit)
    await device.start()
    await asyncio.sleep(0.05)

    assert [name for name, _, _ in events].count("mic.silent_stream") == 1
    assert (await device.health()).status is HealthStatus.DEGRADED
    await device.stop()


async def test_silence_alarm_requires_five_continuous_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_daemon(monkeypatch)
    events: list[str] = []
    zeros = np.zeros(64_000, dtype=np.float32)
    sound = np.full(16_000, 0.05, dtype=np.float32)
    frames = [(16_000, zeros), (16_000, sound), (16_000, zeros)]
    robot = _robot(frames)
    device = _device(robot, emit=lambda name, *args, **kwargs: events.append(name))
    await device.start()
    await asyncio.sleep(0.05)

    assert "mic.silent_stream" not in events
    assert (await device.health()).status is HealthStatus.OK
    await device.stop()


async def test_silence_shorter_than_five_seconds_does_not_alarm(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_daemon(monkeypatch)
    events: list[str] = []
    robot = _robot([(16_000, np.zeros(78_400, dtype=np.float32))])
    device = _device(robot, emit=lambda name, *args, **kwargs: events.append(name))
    await device.start()
    await asyncio.sleep(0.05)

    assert "mic.silent_stream" not in events
    assert (await device.health()).status is HealthStatus.OK
    await device.stop()


async def test_mic_worker_resamples_using_reachy_input_sample_rate(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_daemon(monkeypatch)
    caplog.set_level(logging.INFO, logger="ambient_reachy.device")
    frames = [
        (8_000, np.arange(160, dtype=np.float32) / 160),
        (8_000, np.arange(160, dtype=np.float32) / 160),
    ]
    device = _device(_robot(frames))
    await device.start()

    first = np.frombuffer(await anext(device.mic()), dtype="<i2")
    second = np.frombuffer(await anext(device.mic()), dtype="<i2")

    assert first.size == second.size == 320
    assert caplog.text.count("Resampling Reachy microphone from 8000 Hz to 16000 Hz") == 1
    await device.stop()


async def test_playback_is_nonblocking_ordered_and_validates_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_daemon(monkeypatch)
    robot = _robot()
    device = _device(robot)
    await device.start()

    await device.play(np.array([32767, -32768], dtype="<i2").tobytes())
    await device.play(np.array([1, 2], dtype="<i2").tobytes())
    with pytest.raises(ValueError, match="whole frames"):
        await device.play(b"\0")
    await asyncio.sleep(0.05)
    assert len(robot.media.pushed) == 2
    assert robot.media.pushed[0].tolist() == pytest.approx([32767 / 32768, -1.0])
    assert robot.media.pushed[1].tolist() == pytest.approx([1 / 32768, 2 / 32768])

    await device.play(b"\0\0")
    await device.stop_playback()
    assert robot.media.clears == 1
    assert device._play_queue.empty()
    await device.stop()


async def test_session_updates_are_owned_and_call_robot_in_expected_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_daemon(monkeypatch)
    robot = _robot()
    calls = robot.calls
    hardware = FakeHardware([robot])
    device = ReachyDevice(DeviceContext("reachy-test", {"mic_autorecover": False}), hardware=hardware)
    await device.start()

    await device.on_session(SessionState.wake("mine"))
    await device.on_session(SessionState.active("other", listening=True, speaking=False))
    await device.on_session(SessionState.active("mine", listening=True, speaking=False))
    await device.on_session(SessionState.interrupted("mine"))
    await device.on_session(SessionState.sleep("mine", "done"))

    assert calls[:4] == ["disable_wobbling", "goto_sleep", "enable_motors", "wake_up"]
    assert "manager.start" in hardware.calls
    assert hardware.calls[:3] == ["manager.stop:False", "manager.start", "listening:False"]
    assert "listening:True" in hardware.calls
    assert "speaking:False" in hardware.calls
    assert calls[-2:] == ["disable_wobbling", "goto_sleep"]
    assert "manager.stop:False" in hardware.calls
    await device.stop()


async def test_wake_failure_emits_legacy_robot_event(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_daemon(monkeypatch)
    events: list[tuple[str, str, str, dict[str, object]]] = []

    def emit(name: str, component: str, level: str = "info", **data: object) -> None:
        events.append((name, component, level, data))

    robot = _robot()

    def fail_wake() -> None:
        raise RuntimeError("motor failure")

    robot.wake_up = fail_wake  # type: ignore[method-assign]
    device = _device(robot, emit=emit)
    await device.start()
    await device.on_session(SessionState.wake("mine"))
    await device.on_session(SessionState.sleep("other", "ignored"))

    assert events == [("robot.wake_failed", "live", "error", {"exc_type": "RuntimeError"})]
    await device.stop()


async def test_sleep_failure_emits_legacy_robot_event(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_daemon(monkeypatch)
    events: list[tuple[str, str, str, dict[str, object]]] = []

    def emit(name: str, component: str, level: str = "info", **data: object) -> None:
        events.append((name, component, level, data))

    robot = _robot()

    def fail_sleep() -> None:
        raise RuntimeError("sleep failed")

    robot.goto_sleep = fail_sleep  # type: ignore[method-assign]
    device = _device(robot, emit=emit)
    await device.start()
    await device.on_session(SessionState.wake("mine"))
    await device.on_session(SessionState.sleep("mine", "done"))

    assert ("robot.sleep_failed", "live", "error", {"exc_type": "RuntimeError"}) in events
    await device.stop()


async def test_daemon_loss_marks_down_and_ends_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_daemon(monkeypatch)
    device = _device(_robot())
    await device.start()
    await device._daemon_failed("unreachable")

    assert (await device.health()).reason == "daemon_lost"
    with pytest.raises(RuntimeError, match="not available"):
        await device.play(b"\0\0")
    with pytest.raises(StopAsyncIteration):
        await anext(device.mic())
    await device.stop()


async def test_unavailable_daemon_raises_and_can_be_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    from ambient_reachy import daemon
    from ambient_reachy.device import ReachyUnavailableError

    events: list[tuple[str, str, dict[str, object]]] = []
    attempts = 0

    async def ready(*args: Any, **kwargs: Any) -> bool:
        nonlocal attempts
        del args
        attempts += 1
        if attempts == 1:
            kwargs["emit"]("daemon.not_ready", "daemon", "error", state="unreachable", timeout_s=1.0)
            return False
        return True

    async def watch(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        await asyncio.Event().wait()

    monkeypatch.setattr(daemon, "wait_for_daemon", ready)
    monkeypatch.setattr(daemon, "watch_daemon", watch)
    robot = _robot()
    device = ReachyDevice(
        DeviceContext(
            "reachy-test",
            {"mic_autorecover": False},
            emit=lambda name, component, level="info", **data: events.append((name, level, data)),
        ),
        hardware=FakeHardware([robot]),
    )

    with pytest.raises(ReachyUnavailableError, match="daemon not ready"):
        await device.start()
    assert not device._started
    assert device._robot is None
    assert [name for name, _, _ in events] == ["daemon.not_ready"]

    await device.start()
    await device.stop()
    assert robot.closed


def test_plugin_discovery_and_create_errors() -> None:
    assert "reachy-mini-lite" in available_kinds()
    plugin = load_plugin("reachy-mini-lite")
    assert plugin.kind == "reachy-mini-lite"
    device = create_device(DeviceConfig("reachy-test", "reachy-mini-lite", config={"camera": False}))
    assert device.id == "reachy-test"
    with pytest.raises(DeviceCreateError, match="ValueError"):
        create_device(DeviceConfig("reachy-test", "reachy-mini-lite", config={"unknown": True}))


def test_plugin_import_and_construction_are_lightweight() -> None:
    script = """
import sys
from ambient_device_sdk import DeviceContext
from ambient_device_sdk import available_kinds, load_plugin
assert 'reachy-mini-lite' in available_kinds()
load_plugin('reachy-mini-lite')
from ambient_reachy.plugin import PLUGIN
device = PLUGIN.create(DeviceContext('reachy-test', {'camera': False}))
assert device.id == 'reachy-test'
for name in ('reachy_mini', 'numpy', 'httpx', 'cv2', 'gi', 'openai'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(__file__).parents[3])


async def test_look_tool_requires_callback_and_handles_image_failure() -> None:
    calls: list[tuple[bytes, str]] = []

    async def describe(jpeg: bytes, question: str) -> str:
        calls.append((jpeg, question))
        return "a desk"

    context = DeviceContext("reachy-test", describe_image=describe)
    robot = _robot()
    tool = ReachyDevice(context, hardware=FakeHardware([robot]))
    # The plugin does not offer tools until it is started; the helper itself is dependency-gated.
    from ambient_reachy.look import build_look_tool

    look = build_look_tool(context, robot, camera_enabled=True)
    assert look is not None
    assert await look.handler({"question": "what is visible?"}) == {"description": "a desk"}
    assert calls == [(b"fake-jpeg", "what is visible?")]
    assert build_look_tool(DeviceContext("no-vision"), robot, camera_enabled=True) is None
    assert build_look_tool(context, robot, camera_enabled=False) is None
    assert (await look.handler({"question": ""}))["error"] == "question must be a non-empty string"
    assert tool.tools() == []


async def test_look_tool_reports_vision_failures() -> None:
    events: list[tuple[str, str, str, dict[str, object]]] = []

    async def fail_describe(jpeg: bytes, question: str) -> str:
        del jpeg, question
        raise RuntimeError("vision provider unavailable")

    def emit(name: str, component: str, level: str = "info", **data: object) -> None:
        events.append((name, component, level, data))

    from ambient_reachy.look import build_look_tool

    context = DeviceContext("reachy-test", emit=emit, describe_image=fail_describe)
    look = build_look_tool(context, _robot(), camera_enabled=True)
    assert look is not None
    assert await look.handler({"question": "describe this"}) == {"error": "vision provider unavailable"}
    assert events == [("vision.failed", "vision", "error", {"exc_type": "RuntimeError"})]


async def test_pollen_body_tool_schema_and_handler_are_adapted() -> None:
    calls: list[str] = []
    robot = _robot()
    manager = FakeMovementManager(calls)

    tools = build_body_tools(
        ("head_tracking",),
        robot,
        manager,
        camera_enabled=False,
        motion_duration_s=1.0,
    )

    assert len(tools) == 1
    assert tools[0].name == "head_tracking"
    assert tools[0].parameters["type"] == "object"
    result = await tools[0].handler({"enabled": True})
    assert calls == ["head_tracking:True"]
    assert not result.get("error")
