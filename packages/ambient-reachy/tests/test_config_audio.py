"""Configuration validation and PCM conversion tests."""

import numpy as np
import pytest

from ambient_reachy.audio import resample_linear, frame_to_mono_int16
from ambient_reachy.config import DEFAULT_BODY_TOOLS, ReachyConfig


def test_config_defaults_and_frozen_values() -> None:
    config = ReachyConfig.from_mapping({})

    assert config.daemon_status_url == "http://127.0.0.1:8000/api/daemon/status"
    assert config.body_tools == DEFAULT_BODY_TOOLS
    assert config.motion_duration_s == 1.0
    with pytest.raises(AttributeError):
        config.camera = False  # type: ignore[misc]


@pytest.mark.parametrize(
    "mapping",
    [
        {"unknown": True},
        {"camera": 1},
        {"mic_autorecover": 0},
        {"daemon_wait_s": True},
        {"daemon_wait_s": 0},
        {"daemon_poll_s": float("nan")},
        {"daemon_watchdog_failures": True},
        {"daemon_watchdog_failures": 0},
        {"body_tools": ["not-a-tool"]},
        {"body_tools": ["dance", "dance"]},
        {"body_tools": "dance"},
        {"daemon_status_url": ""},
    ],
)
def test_config_rejects_invalid_values(mapping: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ReachyConfig.from_mapping(mapping)


def test_config_accepts_tool_subsets_and_numeric_values() -> None:
    config = ReachyConfig.from_mapping({"camera": False, "body_tools": ["head_tracking"], "daemon_wait_s": 2})

    assert config.camera is False
    assert config.body_tools == ("head_tracking",)
    assert config.daemon_wait_s == 2.0


def test_frame_conversion_downmixes_and_scales() -> None:
    stereo = np.column_stack((np.full(4, 0.5, dtype=np.float32), np.full(4, -0.25, dtype=np.float32)))

    assert frame_to_mono_int16(stereo).tolist() == [4095] * 4
    assert frame_to_mono_int16(np.array([[1000, -1000], [2000, 2000]], dtype=np.int16)).tolist() == [0, 2000]
    assert frame_to_mono_int16(np.array([0.0, 1.5, -1.5], dtype=np.float32)).tolist() == [0, 32767, -32767]


def test_resample_linear_changes_sample_count_and_preserves_edges() -> None:
    samples = np.array([0, 1000, 2000, 3000], dtype=np.int16)

    converted = resample_linear(samples, 8_000, 16_000)

    assert converted.size == 8
    assert converted[0] == 0
    assert converted[-1] == 3000
    assert resample_linear(samples, 16_000, 16_000) is samples


def test_resample_linear_rejects_invalid_rates() -> None:
    with pytest.raises(ValueError, match="sample rates"):
        resample_linear(np.array([1], dtype=np.int16), 0, 16_000)
