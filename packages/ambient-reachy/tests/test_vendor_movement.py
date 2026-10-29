import time
import threading
from typing import Any

import pytest
from reachy_mini.utils import create_head_pose

from ambient_reachy._vendor.pollen.moves import MovementManager


class FakeRobot:
    def __init__(self) -> None:
        self.targets: list[dict[str, Any]] = []

    def set_target(self, **target: Any) -> None:
        self.targets.append(target)


def test_vendored_movement_manager_lifecycle() -> None:
    robot = FakeRobot()
    manager = MovementManager(current_robot=robot)
    manager.idle_inactivity_delay = 5.0
    manager.start()
    thread = manager._thread
    assert thread is not None

    time.sleep(0.16)
    manager.set_listening(True)
    deadline = time.monotonic() + 1.0
    while not manager._is_listening and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager._is_listening

    manager.set_speaking(True)
    deadline = time.monotonic() + 1.0
    while not manager._command_queue.empty() and time.monotonic() < deadline:
        time.sleep(0.01)

    manager.stop(reset_to_neutral=False)

    assert thread not in threading.enumerate()
    assert manager._thread is None
    assert robot.targets
    assert robot.targets[0]["head"] == pytest.approx(create_head_pose(0, 0, 0, 0, 0, 0, degrees=True))
