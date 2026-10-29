"""Lazy entry point for the Reachy Mini Lite device."""

from typing import TYPE_CHECKING

from ambient_device_sdk import DevicePlugin, DeviceContext


if TYPE_CHECKING:
    from ambient_reachy.device import ReachyDevice


def _create(context: DeviceContext) -> "ReachyDevice":
    from ambient_reachy.device import ReachyDevice

    return ReachyDevice(context)


PLUGIN = DevicePlugin(kind="reachy-mini-lite", create=_create, description="Reachy Mini Lite (USB)")
