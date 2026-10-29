from typing import Any

import numpy as np
import pytest
from reachy_mini.utils import create_head_pose

from ambient_reachy.body_tools import build_body_tools
from ambient_reachy._vendor.pollen.tools import dance as dance_module
from ambient_reachy._vendor.pollen.tools import play_emotion as emotion_module
from ambient_reachy._vendor.pollen.tools.dance import Dance
from ambient_reachy._vendor.pollen.tools.move_head import MoveHead
from ambient_reachy._vendor.pollen.tools.stop_dance import StopDance
from ambient_reachy._vendor.pollen.tools.sweep_look import SweepLook
from ambient_reachy._vendor.pollen.tools.play_emotion import PlayEmotion
from ambient_reachy._vendor.pollen.tools.stop_emotion import StopEmotion
from ambient_reachy._vendor.pollen.dance_emotion_moves import GotoQueueMove, DanceQueueMove, EmotionQueueMove
from ambient_reachy._vendor.pollen.tools.head_tracking import HeadTracking


class FakeRobot:
    def get_current_head_pose(self) -> np.ndarray:
        return create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)

    def get_current_joint_positions(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(6), np.zeros(2)


class FakeMovementManager:
    def __init__(self) -> None:
        self.queued: list[Any] = []
        self.clears = 0
        self.moving_states: list[float] = []
        self.tracking: list[bool] = []

    def queue_move(self, move: Any) -> None:
        self.queued.append(move)

    def clear_move_queue(self) -> None:
        self.clears += 1
        self.queued.clear()

    def set_moving_state(self, duration: float) -> None:
        self.moving_states.append(duration)

    def set_head_tracking(self, enabled: bool) -> None:
        self.tracking.append(enabled)


class FakeRecordedMove:
    duration = 1.0

    def evaluate(self, time_s: float) -> tuple[None, np.ndarray, float]:
        del time_s
        return None, np.zeros(2), 0.0


class FakeRecordedMoves:
    def __init__(self, dataset: str) -> None:
        assert dataset == "pollen-robotics/reachy-mini-emotions-library"

    def list_moves(self) -> list[str]:
        return ["happy1"]

    def get(self, name: str) -> FakeRecordedMove:
        assert name == "happy1"
        return FakeRecordedMove()


@pytest.mark.asyncio
async def test_all_vendored_tools_adapt_specs_and_execute_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emotion_module, "RecordedMoves", FakeRecordedMoves)
    names = (
        "dance",
        "stop_dance",
        "play_emotion",
        "stop_emotion",
        "move_head",
        "sweep_look",
        "head_tracking",
    )
    tools = build_body_tools(names, FakeRobot(), FakeMovementManager(), camera_enabled=False, motion_duration_s=1.25)
    specs = dict(zip(names, tools, strict=True))
    tool_classes = (Dance, StopDance, PlayEmotion, StopEmotion, MoveHead, SweepLook, HeadTracking)

    assert tuple(specs) == names
    for name, tool_class in zip(names, tool_classes, strict=True):
        spec = specs[name]
        assert spec.schema()["name"] == tool_class.name
        assert spec.description == tool_class.description
        assert spec.parameters == tool_class.parameters_schema
        assert spec.parameters["type"] == "object"

    manager = FakeMovementManager()
    robot = FakeRobot()
    tools = build_body_tools(names, robot, manager, camera_enabled=False, motion_duration_s=1.25)
    specs = dict(zip(names, tools, strict=True))
    move_name = next(iter(dance_module.AVAILABLE_MOVES))

    result = await specs["dance"].handler({"move": move_name, "repeat": 1})
    assert result["status"] == "queued"
    assert isinstance(manager.queued[-1], DanceQueueMove)

    await specs["stop_dance"].handler({"dummy": True})
    assert manager.clears == 1

    result = await specs["play_emotion"].handler({"emotion": "random"})
    assert result["status"] == "queued"
    assert isinstance(manager.queued[-1], EmotionQueueMove)

    await specs["stop_emotion"].handler({"dummy": True})
    assert manager.clears == 2

    result = await specs["move_head"].handler({"direction": "left"})
    assert result["status"] == "looking left"
    assert isinstance(manager.queued[-1], GotoQueueMove)
    assert manager.moving_states[-1] == 1.25

    result = await specs["sweep_look"].handler({})
    assert result["status"].startswith("sweeping look")
    assert len(manager.queued) == 6
    assert manager.moving_states[-1] == 14.0

    result = await specs["head_tracking"].handler({"enabled": True})
    assert result["status"] == "following"
    assert manager.tracking == [True]
