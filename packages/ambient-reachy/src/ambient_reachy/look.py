"""Vision tool that delegates image description to the ambient-home hub."""

import asyncio
import logging
from typing import cast

from ambient_device_sdk import ToolSpec, DeviceContext


logger = logging.getLogger(__name__)


LOOK_DESCRIPTION = "Use the camera and vision model to answer a question about what is visible."
LOOK_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
}


def build_look_tool(context: DeviceContext, robot: object, *, camera_enabled: bool) -> ToolSpec | None:
    """Build the look tool when camera and the hub vision callback are available."""
    if not camera_enabled or context.describe_image is None:
        return None
    describe_image = context.describe_image

    async def handler(args: dict[str, object]) -> dict[str, object]:
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            return {"error": "question must be a non-empty string"}
        if not camera_enabled:
            return {"error": "Camera is disabled"}
        try:
            media = getattr(robot, "media")
            jpeg = await asyncio.to_thread(media.get_frame_jpeg)
            if not jpeg:
                return {"error": "No frame available"}
            answer = await describe_image(cast(bytes, jpeg), question)
            return {"description": answer}
        except Exception as exc:
            logger.exception("Failed to describe Reachy camera image")
            context.emit("vision.failed", "vision", "error", exc_type=type(exc).__name__)
            return {"error": str(exc)}

    return ToolSpec("look", LOOK_DESCRIPTION, LOOK_PARAMETERS, handler)
