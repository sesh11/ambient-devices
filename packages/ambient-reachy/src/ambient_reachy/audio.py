"""Audio conversion helpers for Reachy Mini media frames."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def frame_to_mono_int16(audio: NDArray[np.generic]) -> NDArray[np.int16]:
    """Downmix a Reachy audio frame and convert it to signed 16-bit samples."""
    samples = np.asarray(audio)
    if samples.ndim not in (1, 2):
        raise ValueError("audio frames must be one- or two-dimensional")
    integer_input = np.issubdtype(samples.dtype, np.integer)
    if samples.ndim == 2:
        if samples.shape[0] < samples.shape[1]:
            samples = samples.T
        samples = np.mean(samples, axis=1)
    if integer_input:
        return np.asarray(np.clip(samples, -32768, 32767), dtype=np.int16)
    return np.asarray(np.clip(samples, -1.0, 1.0) * 32767.0, dtype=np.int16)


def resample_linear(samples: NDArray[np.int16], source_rate: int, target_rate: int) -> NDArray[np.int16]:
    """Resample mono samples using linear interpolation."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate or samples.size == 0:
        return samples
    output_size = max(1, round(samples.size * target_rate / source_rate))
    positions = np.linspace(0, samples.size - 1, output_size)
    return np.asarray(np.interp(positions, np.arange(samples.size), samples), dtype=np.int16)
