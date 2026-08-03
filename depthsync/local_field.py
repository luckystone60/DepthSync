"""Compact local affine fields for DepthSync V5 playback."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .flow import DenseFlowSequence


FIELD_DELTA_SCALE_BOUNDS = (-0.5, 0.5)
# FP16 represents 0.35 as 0.35009766.  The narrow margin is part of the
# serialized contract; playback still clamps to LocalFieldConfig's ±0.35.
FIELD_OFFSET_NORM_BOUNDS = (-0.351, 0.351)


@dataclass(frozen=True)
class LocalFieldConfig:
    grid_shape: tuple[int, int] = (36, 64)
    scale_bounds: tuple[float, float] = (0.5, 1.5)
    offset_bound_fraction: float = 0.35
    min_fit_pixels: int = 24
    scale_ridge: float = 1e-2
    offset_ridge: float = 1e-3
    flow_confidence_low: float = 0.05
    flow_confidence_high: float = 0.25
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
        if any(
            not np.all(np.isfinite(channel))
            for channel in (
                self.delta_scale,
                self.offset_norm,
                self.confidence,
                self.depth_low,
            )
        ):
            raise ValueError("local field channels must be finite")
        if np.any((self.confidence < 0.0) | (self.confidence > 1.0)):
            raise ValueError("local field confidence must be within [0,1]")
        if np.any(
            (self.delta_scale < FIELD_DELTA_SCALE_BOUNDS[0])
            | (self.delta_scale > FIELD_DELTA_SCALE_BOUNDS[1])
        ):
            raise ValueError("local field scale is outside the serialized contract")
        if np.any(
            (self.offset_norm < FIELD_OFFSET_NORM_BOUNDS[0])
            | (self.offset_norm > FIELD_OFFSET_NORM_BOUNDS[1])
        ):
            raise ValueError("local field offset is outside the serialized contract")

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
    if any(
        not np.all(np.isfinite(channel))
        for channel in (
            field.delta_scale,
            field.offset_norm,
            field.confidence,
            field.depth_low,
        )
    ):
        raise ValueError("local field channels must be finite")
    if np.any((field.confidence < 0.0) | (field.confidence > 1.0)):
        raise ValueError("local field confidence must be within [0,1]")
    if np.any(
        (field.delta_scale < FIELD_DELTA_SCALE_BOUNDS[0])
        | (field.delta_scale > FIELD_DELTA_SCALE_BOUNDS[1])
    ):
        raise ValueError("local field scale is outside the serialized contract")
    if np.any(
        (field.offset_norm < FIELD_OFFSET_NORM_BOUNDS[0])
        | (field.offset_norm > FIELD_OFFSET_NORM_BOUNDS[1])
    ):
        raise ValueError("local field offset is outside the serialized contract")


def _depth_guided_upsample(
    values: tuple[np.ndarray, ...],
    confidence: np.ndarray,
    depth_low: np.ndarray,
    base: np.ndarray,
    depth_sigma: float,
    eps: float,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Jointly upsample a coarse field without mixing across depth edges."""
    height, width = base.shape
    gh, gw = confidence.shape
    grid_x = (np.arange(width, dtype=np.float32) + 0.5) * gw / width - 0.5
    grid_y = (np.arange(height, dtype=np.float32) + 0.5) * gh / height - 0.5
    x0_raw = np.floor(grid_x).astype(np.int32)
    y0_raw = np.floor(grid_y).astype(np.int32)
    tx = grid_x - x0_raw
    ty = grid_y - y0_raw
    x_indices = (np.clip(x0_raw, 0, gw - 1), np.clip(x0_raw + 1, 0, gw - 1))
    y_indices = (np.clip(y0_raw, 0, gh - 1), np.clip(y0_raw + 1, 0, gh - 1))
    x_weights = (1.0 - tx, tx)
    y_weights = (1.0 - ty, ty)

    guide_denominator = np.zeros((height, width), np.float32)
    supported_denominator = np.zeros((height, width), np.float32)
    numerators = [np.zeros((height, width), np.float32) for _ in values]
    safe_base = np.nan_to_num(base, nan=0.0).astype(np.float32)
    sigma = max(float(depth_sigma), eps)
    for yi, wy in zip(y_indices, y_weights):
        for xi, wx in zip(x_indices, x_weights):
            spatial = wy[:, None] * wx[None, :]
            node_depth = depth_low[yi[:, None], xi[None, :]]
            depth_weight = np.exp(
                np.clip(-np.abs(safe_base - node_depth) / sigma, -50.0, 0.0)
            )
            guide_weight = spatial * depth_weight
            node_confidence = confidence[yi[:, None], xi[None, :]]
            supported_weight = guide_weight * node_confidence
            guide_denominator += guide_weight
            supported_denominator += supported_weight
            for numerator, value in zip(numerators, values):
                numerator += supported_weight * value[yi[:, None], xi[None, :]]

    upsampled_values = tuple(
        np.where(
            supported_denominator > eps,
            numerator / np.maximum(supported_denominator, eps),
            0.0,
        ).astype(np.float32)
        for numerator in numerators
    )
    upsampled_confidence = np.where(
        guide_denominator > eps,
        supported_denominator / np.maximum(guide_denominator, eps),
        0.0,
    )
    return upsampled_values, np.clip(upsampled_confidence, 0.0, 1.0).astype(np.float32)


def apply_local_field(
    base: np.ndarray,
    field: LocalFieldFrame,
    config: LocalFieldConfig,
) -> np.ndarray:
    """Apply one field, preserving V4 bit-exactly when confidence is zero."""
    base = np.asarray(base, dtype=np.float32)
    if base.ndim != 2:
        raise ValueError("base must have shape [H,W]")
    _validate_frame(field)
    confidence = np.asarray(field.confidence, dtype=np.float32)
    if float(np.max(confidence, initial=0.0)) <= 0.0:
        return base.copy()
    (delta_scale, offset_norm), weight = _depth_guided_upsample(
        (
            np.asarray(field.delta_scale, dtype=np.float32),
            np.asarray(field.offset_norm, dtype=np.float32),
        ),
        np.clip(confidence, 0.0, 1.0).astype(np.float32),
        np.asarray(field.depth_low, dtype=np.float32),
        base,
        # ``base`` and ``depth_low`` are in the V4 working-depth domain.  The
        # photo anchor may have a very different raw range (notably DAv2), so
        # using ``photo_range`` here silently turns the edge guard into a
        # near-uniform bilinear blend.  This threshold must stay in the guide
        # domain.
        config.upsample_depth_sigma_fraction * _guide_range(base, config.eps),
        config.eps,
    )
    scale = np.clip(
        1.0 + delta_scale,
        config.scale_bounds[0],
        config.scale_bounds[1],
    )
    offset_norm = np.clip(
        offset_norm,
        -config.offset_bound_fraction,
        config.offset_bound_fraction,
    )
    correction = (scale - 1.0) * base + offset_norm * field.photo_range
    return (base + weight * correction).astype(np.float32)


def _photo_range(photo: np.ndarray, eps: float) -> float:
    finite = photo[np.isfinite(photo)]
    if finite.size == 0:
        return 1.0
    value = float(np.quantile(finite, 0.9) - np.quantile(finite, 0.1))
    return max(value, eps)


def _guide_range(depth: np.ndarray, eps: float) -> float:
    """Robust guide-domain range with a FP16-safe floor for flat regions."""
    finite = np.asarray(depth, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return eps
    magnitude = float(np.quantile(np.abs(finite), 0.9))
    return max(_photo_range(finite, eps), 0.01 * magnitude, eps)


def _odd_window(value: float) -> int:
    result = max(9, int(round(value)))
    return result if result % 2 else result + 1


def _smooth_supported(values: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    kernel = np.asarray([1.0, 2.0, 1.0], np.float32)
    weighted = values * confidence
    numerator = cv2.sepFilter2D(weighted, -1, kernel, kernel)
    denominator = cv2.sepFilter2D(confidence, -1, kernel, kernel)
    smoothed = numerator / np.maximum(denominator, 1e-6)
    return np.where(confidence > 0, smoothed, 0.0).astype(np.float32)


def fit_anchor_field(
    base_anchor: np.ndarray,
    photo_anchor: np.ndarray,
    config: LocalFieldConfig,
) -> LocalFieldFrame:
    """Fit overlapping robust local affine models relative to a V4 anchor."""
    base = np.asarray(base_anchor, dtype=np.float32)
    photo = np.asarray(photo_anchor, dtype=np.float32)
    if base.ndim != 2 or photo.shape != base.shape:
        raise ValueError("base_anchor and photo_anchor must have equal [H,W] shapes")
    gh, gw = config.grid_shape
    if gh <= 0 or gw <= 0:
        raise ValueError("grid_shape must be positive")
    height, width = base.shape
    photo_range = _photo_range(photo, config.eps)
    scale = np.ones((gh, gw), np.float32)
    offset = np.zeros((gh, gw), np.float32)
    confidence = np.zeros((gh, gw), np.float32)
    window_h = _odd_window(2.5 * height / gh)
    window_w = _odd_window(2.5 * width / gw)
    centers_y = (np.arange(gh, dtype=np.float32) + 0.5) * height / gh
    centers_x = (np.arange(gw, dtype=np.float32) + 0.5) * width / gw

    for gy, center_y in enumerate(centers_y):
        y0 = max(0, int(round(center_y)) - window_h // 2)
        y1 = min(height, y0 + window_h)
        y0 = max(0, y1 - window_h)
        for gx, center_x in enumerate(centers_x):
            x0 = max(0, int(round(center_x)) - window_w // 2)
            x1 = min(width, x0 + window_w)
            x0 = max(0, x1 - window_w)
            source = base[y0:y1, x0:x1].reshape(-1)
            target = photo[y0:y1, x0:x1].reshape(-1)
            valid = np.isfinite(source) & np.isfinite(target)
            count = int(np.count_nonzero(valid))
            if count < config.min_fit_pixels:
                continue
            source = source[valid].astype(np.float64)
            target = target[valid].astype(np.float64)
            design = np.column_stack((source, np.ones_like(source)))
            weights = np.ones(count, np.float64)
            solution = np.asarray([1.0, 0.0], np.float64)
            lhs = np.eye(2, dtype=np.float64)
            for _ in range(2):
                lhs = design.T @ (weights[:, None] * design)
                lhs += np.diag([config.scale_ridge, config.offset_ridge])
                rhs = design.T @ (weights * target)
                rhs[0] += config.scale_ridge
                solution = np.linalg.solve(lhs, rhs)
                residual = target - design @ solution
                median = float(np.median(residual))
                mad = float(np.median(np.abs(residual - median)))
                robust_scale = max(1.4826 * mad, 0.01 * photo_range, config.eps)
                normalized = np.abs(residual - median) / (4.685 * robust_scale)
                weights = np.square(np.clip(1.0 - normalized**2, 0.0, 1.0))
            fitted_scale = float(
                np.clip(solution[0], config.scale_bounds[0], config.scale_bounds[1])
            )
            fitted_offset = float(
                np.clip(
                    solution[1],
                    -config.offset_bound_fraction * photo_range,
                    config.offset_bound_fraction * photo_range,
                )
            )
            # Re-optimize after box constraints. A low-variance window can put
            # the entire correction in the offset and then lose it to clipping;
            # alternating the two bounded coordinates preserves the best fit.
            for _ in range(2):
                scale_denominator = float(
                    np.sum(weights * source * source) + config.scale_ridge
                )
                fitted_scale = float(
                    np.clip(
                        (
                            np.sum(weights * source * (target - fitted_offset))
                            + config.scale_ridge
                        )
                        / max(scale_denominator, config.eps),
                        config.scale_bounds[0],
                        config.scale_bounds[1],
                    )
                )
                fitted_offset = float(
                    np.clip(
                        np.median(target - fitted_scale * source),
                        -config.offset_bound_fraction * photo_range,
                        config.offset_bound_fraction * photo_range,
                    )
                )
            residual = target - (fitted_scale * source + fitted_offset)
            support = min(1.0, count / float(window_h * window_w))
            residual_score = np.exp(
                -float(np.median(np.abs(residual))) / (0.08 * photo_range + config.eps)
            )
            condition = float(np.linalg.cond(lhs))
            condition_score = 1.0 / (1.0 + max(np.log10(max(condition, 1.0)), 0.0) / 12.0)
            scale[gy, gx] = fitted_scale
            offset[gy, gx] = fitted_offset
            confidence[gy, gx] = float(
                np.clip(support * residual_score * condition_score, 0.0, 1.0)
            )

    delta_scale = _smooth_supported(scale - 1.0, confidence)
    offset_norm = _smooth_supported(offset / photo_range, confidence)
    depth_low = cv2.resize(base, (gw, gh), interpolation=cv2.INTER_AREA)
    return LocalFieldFrame(
        delta_scale=delta_scale,
        offset_norm=offset_norm,
        confidence=confidence,
        depth_low=depth_low.astype(np.float32),
        photo_range=photo_range,
    )


def _resize_flow_to_grid(
    flow: np.ndarray,
    confidence: np.ndarray,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    source_h, source_w = flow.shape[:2]
    grid_h, grid_w = grid_shape
    resized = cv2.resize(
        flow.astype(np.float32),
        (grid_w, grid_h),
        interpolation=cv2.INTER_LINEAR,
    )
    resized[..., 0] *= grid_w / max(source_w, 1)
    resized[..., 1] *= grid_h / max(source_h, 1)
    resized_confidence = cv2.resize(
        confidence.astype(np.float32),
        (grid_w, grid_h),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized, np.clip(resized_confidence, 0.0, 1.0)


def _remap_grid(values: np.ndarray, current_to_previous: np.ndarray) -> np.ndarray:
    height, width = current_to_previous.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    return cv2.remap(
        values.astype(np.float32),
        xx + current_to_previous[..., 0],
        yy + current_to_previous[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def propagate_local_fields(
    base_depths: np.ndarray,
    rgb_frames: np.ndarray,
    anchor_field: LocalFieldFrame,
    anchor_index: int,
    flow: DenseFlowSequence,
    config: LocalFieldConfig,
) -> LocalFieldSequence:
    """Propagate an anchor parameter field along trusted adjacent flow."""
    base = np.asarray(base_depths, dtype=np.float32)
    rgb = np.asarray(rgb_frames)
    if base.ndim != 3 or rgb.ndim != 4 or len(base) != len(rgb):
        raise ValueError("base_depths and rgb_frames must have equal frame counts")
    frame_count = len(base)
    if not 0 <= anchor_index < frame_count:
        raise ValueError("anchor_index is out of range")
    if flow.to_next.shape[0] != frame_count - 1:
        raise ValueError("flow must contain T-1 adjacent pairs")
    gh, gw = config.grid_shape
    _validate_frame(anchor_field)
    if anchor_field.delta_scale.shape != (gh, gw):
        raise ValueError("anchor field shape must equal config.grid_shape")

    depth_low = np.stack(
        [
            cv2.resize(frame, (gw, gh), interpolation=cv2.INTER_AREA)
            for frame in base
        ]
    ).astype(np.float32)
    q = np.zeros((frame_count, gh, gw, 2), np.float32)
    confidence = np.zeros((frame_count, gh, gw), np.float32)
    q[anchor_index, ..., 0] = anchor_field.delta_scale
    q[anchor_index, ..., 1] = anchor_field.offset_norm
    confidence[anchor_index] = np.clip(anchor_field.confidence, 0.0, 1.0)

    for direction in (-1, 1):
        previous_index = anchor_index
        stop = -1 if direction < 0 else frame_count
        for index in range(anchor_index + direction, stop, direction):
            pair = index if direction < 0 else index - 1
            if direction < 0:
                current_to_previous, flow_confidence = _resize_flow_to_grid(
                    flow.to_next[pair], flow.confidence_next[pair], (gh, gw)
                )
            else:
                current_to_previous, flow_confidence = _resize_flow_to_grid(
                    flow.to_previous[pair],
                    flow.confidence_previous[pair],
                    (gh, gw),
                )
            if flow.scene_cuts[pair]:
                flow_confidence = np.zeros_like(flow_confidence)
            warped_q = _remap_grid(q[previous_index], current_to_previous)
            warped_confidence = _remap_grid(
                confidence[previous_index], current_to_previous
            )
            visibility = np.clip(
                (flow_confidence - config.flow_confidence_low)
                / max(
                    config.flow_confidence_high - config.flow_confidence_low,
                    config.eps,
                ),
                0.0,
                1.0,
            )
            visibility = visibility * visibility * (3.0 - 2.0 * visibility)
            data_confidence = np.clip(
                warped_confidence * visibility,
                0.0,
                1.0,
            )
            active = data_confidence > config.eps
            # The anchor fit is regularized once. Re-regularizing on each RGB
            # frame rewrites a field that is already aligned by flow and causes
            # temporal breathing. Trusted trajectories transport it unchanged.
            q[index] = warped_q
            q[index, ~active] = 0.0
            confidence[index] = np.where(active, data_confidence, 0.0)
            previous_index = index

    q[..., 0] = np.clip(
        q[..., 0], config.scale_bounds[0] - 1.0, config.scale_bounds[1] - 1.0
    )
    q[..., 1] = np.clip(
        q[..., 1],
        -config.offset_bound_fraction,
        config.offset_bound_fraction,
    )
    return LocalFieldSequence(
        delta_scale=q[..., 0],
        offset_norm=q[..., 1],
        confidence=confidence,
        depth_low=depth_low,
        photo_range=anchor_field.photo_range,
    )


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
