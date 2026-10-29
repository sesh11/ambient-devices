"""Adapters from Pollen conversation tools to ambient-device-sdk ToolSpecs."""

import logging
from typing import Any, cast

from ambient_device_sdk import ToolSpec


logger = logging.getLogger(__name__)


def build_body_tools(
    names: tuple[str, ...],
    robot: object,
    movement_manager: object,
    *,
    camera_enabled: bool,
    motion_duration_s: float,
) -> list[ToolSpec]:
    """Instantiate configured Pollen tool classes directly and adapt their handlers."""
    from reachy_mini_conversation_app.tools.dance import Dance
    from reachy_mini_conversation_app.tools.move_head import MoveHead
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.tools.stop_dance import StopDance
    from reachy_mini_conversation_app.tools.sweep_look import SweepLook
    from reachy_mini_conversation_app.tools.play_emotion import PlayEmotion
    from reachy_mini_conversation_app.tools.stop_emotion import StopEmotion
    from reachy_mini_conversation_app.tools.head_tracking import HeadTracking

    classes: dict[str, type[Any]] = {
        "dance": Dance,
        "stop_dance": StopDance,
        "play_emotion": PlayEmotion,
        "stop_emotion": StopEmotion,
        "move_head": MoveHead,
        "sweep_look": SweepLook,
        "head_tracking": HeadTracking,
    }
    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_enabled=camera_enabled,
        motion_duration_s=motion_duration_s,
    )
    result: list[ToolSpec] = []
    for name in names:
        tool = classes[name]()

        async def handler(args: dict[str, object], pollen_tool: Any = tool) -> dict[str, object]:
            try:
                return cast(dict[str, object], await pollen_tool(deps, **args))
            except Exception as exc:
                logger.exception("Pollen device tool failed: %s", pollen_tool.name)
                return {"error": str(exc)}

        result.append(ToolSpec(tool.name, tool.description, tool.parameters_schema, handler))
    return result
