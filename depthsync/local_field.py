"""Compact local affine fields for DepthSync V5 playback."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LocalFieldConfig:
    grid_shape: tuple[int, int] = (36, 64)
    scale_bounds: tuple[float, float] = (0.5, 1.5)
    offset_bound_fraction: float = 0.35
    min_fit_pixels: int = 24
    scale_ridge: float = 1e-2
    offset_ridge: float = 1e-3
    spatial_iterations: int = 5
    rgb_sigma: float = 24.0
    depth_sigma_fraction: float = 0.04
    spatial_weight: float = 2.0
    identity_weight: float = 1.0
    temporal_smoothing: float = 0.65
    max_scale_step: float = 0.03
    max_offset_step_fraction: float = 0.02
    upsample_depth_sigma_fraction: float = 0.04
    eps: float = 1e-6


@dataclass(frozen=True)
class LocalFieldFrame:
    delta_scale: np.ndarray
    offset_norm: np.ndarray
    confidence: np.ndarray
    depth_low: np.ndarray
    photo_range: float


@dataclass(frozen=True)
class LocalFieldSequence:
    delta_scale: np.ndarray
    offset_norm: np.ndarray
    confidence: np.ndarray
    depth_low: np.ndarray
    photo_range: float

    def __post_init__(self) -> None:
        shape = self.delta_scale.shape
        if len(shape) != 3:
            raise ValueError("local field channels must have shape [T,Gh,Gw]")
        if any(
            channel.shape != shape
            for channel in (self.offset_norm, self.confidence, self.depth_low)
        ):
            raise ValueError("local field channels must have equal shapes")
        if not np.isfinite(self.photo_range) or self.photo_range <= 0:
            raise ValueError("photo_range must be finite and positive")

    def frame(self, index: int) -> LocalFieldFrame:
        return LocalFieldFrame(
            self.delta_scale[index],
            self.offset_norm[index],
            self.confidence[index],
            self.depth_low[index],
            self.photo_range,
        )


def _validate_frame(field: LocalFieldFrame) -> None:
    shape = field.delta_scale.shape
    if len(shape) != 2 or any(
        channel.shape != shape
        for channel in (field.offset_norm, field.confidence, field.depth_low)
    ):
        raise ValueError("local field frame channels must have equal [Gh,Gw] shapes")
    if not np.isfinite(field.photo_range) or field.photo_range <= 0:
        raise ValueError("photo_range must be finite and positive")


def apply_local_field(
    base: np.ndarray,
    field: LocalFieldFrame,
    config: LocalFieldConfig,
) -> np.ndarray:
    """Apply one field, preserving V4 bit-exactly when confidence is zero."""
    del config
    base = np.asarray(base, dtype=np.float32)
    if base.ndim != 2:
        raise ValueError("base must have shape [H,W]")
    _validate_frame(field)
    confidence = np.nan_to_num(field.confidence, nan=0.0)
    if float(np.max(confidence, initial=0.0)) <= 0.0:
        return base.copy()
    raise NotImplementedError("nonzero local-field playback is added with anchor fitting")


def save_local_fields(path: Path | str, fields: LocalFieldSequence) -> None:
    """Save the prototype field cache using the mobile FP16 channel format."""
    np.savez_compressed(
        Path(path),
        delta_scale=fields.delta_scale.astype(np.float16),
        offset_norm=fields.offset_norm.astype(np.float16),
        confidence=fields.confidence.astype(np.float16),
        depth_low=fields.depth_low.astype(np.float16),
        photo_range=np.float64(fields.photo_range),
    )


def load_local_fields(path: Path | str) -> LocalFieldSequence:
    """Load and validate a prototype field cache."""
    with np.load(Path(path), allow_pickle=False) as payload:
        return LocalFieldSequence(
            delta_scale=payload["delta_scale"].astype(np.float16, copy=True),
            offset_norm=payload["offset_norm"].astype(np.float16, copy=True),
            confidence=payload["confidence"].astype(np.float16, copy=True),
            depth_low=payload["depth_low"].astype(np.float16, copy=True),
            photo_range=float(payload["photo_range"]),
        )
