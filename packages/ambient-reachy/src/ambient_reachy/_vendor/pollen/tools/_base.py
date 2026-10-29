# Apache-2.0; derived from reachy_mini_conversation_app/tools/core_tools.py at f885bf421764ff7ff18dff26ae77b24b99a3af8c.
# Removed configuration, MCP, tool-space, profile, dynamic-loading, and registry code.

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Literal, TypedDict

from reachy_mini import ReachyMini


@dataclass
class ToolDependencies:
    """External dependencies injected into tools."""

    reachy_mini: ReachyMini
    movement_manager: Any  # MovementManager from moves.py
    # Optional deps
    instance_path: str | Path | None = None
    camera_enabled: bool = False
    motion_duration_s: float = 1.0
    go_to_sleep: Callable[[], dict[str, Any]] | None = None


class ToolSpec(TypedDict):
    """Function-calling spec for a tool, in the OpenAI-compatible shape."""

    type: Literal["function"]
    name: str
    description: str
    parameters: dict[str, Any]  # arbitrary JSON Schema


class Tool(abc.ABC):
    """Base abstraction for tools used in function-calling.

    Each tool must define:
      - name: str
      - description: str
      - parameters_schema: Dict[str, Any]  # JSON Schema

    Tools may override:
      - needs_response: bool = True  # set False to skip the spoken follow-up after this tool runs
    """

    _auto_register: ClassVar[bool] = True
    needs_response: ClassVar[bool] = True

    name: str
    description: str
    parameters_schema: Dict[str, Any]

    def spec(self) -> ToolSpec:
        """Return the function spec for LLM consumption."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    @abc.abstractmethod
    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Async tool execution entrypoint."""
        raise NotImplementedError
