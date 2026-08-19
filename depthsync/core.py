"""Parameter-only photo-anchor depth synchronization for edge deployment."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import cv2
import numpy as np

@dataclass(frozen=True)
class DepthSyncConfig:
    """Configuration for the lightweight parameter-estimation path."""

    depth_mode: str = "disparity"
    sample_count: int = 16384
    face_sample_fraction: float = 0.30
    edge_quantile: float = 0.70
    mad_threshold: float = 3.0
    parameter_smoothing: float = 0.65
    anchor_lock_radius: int = 1
    min_fit_pixels: int = 128
    high_res_interpolation: int = cv2.INTER_LINEAR
    lut_nodes: int = 64
    lut_candidate_nodes: tuple[int, ...] = (8, 16, 32, 64)
    lut_quantile_low: float = 0.02
    lut_quantile_high: float = 0.98
    lut_distribution_weight: float = 0.50
    lut_complexity_penalty: float = 0.002
    lut_min_pair_correlation: float = 0.05
    lut_temporal_nodes: int = 16
    lut_distribution_blend: float = 1.0
    lut_distribution_clip_fraction: float = 0.15
    lut_frame_min_pair_correlation: float = 0.80
    lut_frame_force_jump_threshold: float = 0.10
    lut_frame_freeze_jump_threshold: float = 0.05
    invalid_floor_quantile: float = 0.10
    invalid_floor_fraction: float = 0.03
    invalid_floor_tolerance_fraction: float = 0.002
    anchor_transition_radius: int = 6
    lut_max_scale_delta: float = 0.01
    lut_max_offset_delta_fraction: float = 0.005
    enable_lut_distribution_stabilization: bool = True
    enable_lut_temporal_adjustment: bool = False
    subject_offset_clip_fraction: float = 0.10
    eps: float = 1e-6


@dataclass(frozen=True)
class MotionSequence:
    """Normalized block motion fields.

    ``to_previous[i]`` maps pixels in frame i to frame i-1. ``to_next[i]``
    maps pixels in frame i to frame i+1. Displacements are normalized by image
    width/height, so the grids can be much smaller than the depth maps.
    """

    to_previous: np.ndarray  # [T,Gh,Gw,2]
    to_next: np.ndarray  # [T,Gh,Gw,2]
    confidence_previous: Optional[np.ndarray] = None  # [T,Gh,Gw]
    confidence_next: Optional[np.ndarray] = None  # [T,Gh,Gw]


@dataclass(frozen=True)
class FrameParameters:
    scale: float
    offset: float
    confidence: float
    fallback_reason: str = ""
    lut_x: Optional[np.ndarray] = None
    lut_y: Optional[np.ndarray] = None


@dataclass
class SyncResult:
    depths: np.ndarray
    scales: np.ndarray
    offsets: np.ndarray
    confidences: np.ndarray
    fallback_reasons: tuple[str, ...]
    lut_x: np.ndarray
    lut_y: np.ndarray

    @property
    def parameters(self) -> tuple[FrameParameters, ...]:
        return tuple(
            FrameParameters(
                float(a),
                float(b),
                float(c),
                reason,
                x,
                y,
            )
            for a, b, c, reason, x, y in zip(
                self.scales,
                self.offsets,
                self.confidences,
                self.fallback_reasons,
                self.lut_x,
                self.lut_y,
            )
        )


def _valid(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x)


def _resize(x: np.ndarray, shape: tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(x.astype(np.float32), (shape[1], shape[0]), interpolation=interpolation)


def _to_working(x: np.ndarray, mode: str, eps: float) -> np.ndarray:
    x = x.astype(np.float32)
    if mode == "disparity":
        return np.where(np.isfinite(x), x, np.nan)
    if mode == "depth":
        return np.where(np.isfinite(x) & (x > 0), 1.0 / np.maximum(x, eps), np.nan)
    raise ValueError("depth_mode must be 'depth' or 'disparity'")


def _from_working(x: np.ndarray, mode: str, eps: float) -> np.ndarray:
    if mode == "depth":
        return np.where(np.isfinite(x) & (x > eps), 1.0 / np.maximum(x, eps), np.nan)
    return np.maximum(x, 0.0)


def _weighted_affine(x: np.ndarray, y: np.ndarray, w: np.ndarray, eps: float) -> tuple[float, float]:
    sw = float(np.sum(w)) + eps
    mx, my = float(np.sum(w * x) / sw), float(np.sum(w * y) / sw)
    var = float(np.sum(w * (x - mx) ** 2)) + eps
    scale = float(np.sum(w * (x - mx) * (y - my)) / var)
    return scale, my - scale * mx


def robust_affine_samples(
    source: np.ndarray,
    target: np.ndarray,
    weights: Optional[np.ndarray],
    cfg: DepthSyncConfig,
) -> tuple[float, float, float]:
    """Two-pass robust affine fit over a bounded set of samples."""
    good = np.isfinite(source) & np.isfinite(target)
    if weights is not None:
        good &= np.isfinite(weights) & (weights > 0)
    x, y = source[good].astype(np.float64), target[good].astype(np.float64)
    w = np.ones_like(x) if weights is None else weights[good].astype(np.float64)
    if x.size < cfg.min_fit_pixels or np.std(x) < cfg.eps:
        return 1.0, 0.0, 0.0

    qx = np.quantile(x, [0.1, 0.5, 0.9])
    qy = np.quantile(y, [0.1, 0.5, 0.9])
    scale = max(float((qy[2] - qy[0]) / max(qx[2] - qx[0], cfg.eps)), cfg.eps)
    offset = float(qy[1] - scale * qx[1])
    residual = y - (scale * x + offset)
    center = float(np.median(residual))
    sigma = 1.4826 * float(np.median(np.abs(residual - center))) + cfg.eps
    inlier = np.abs(residual - center) <= cfg.mad_threshold * sigma
    if int(np.count_nonzero(inlier)) < cfg.min_fit_pixels:
        return scale, offset, 0.0

    scale, offset = _weighted_affine(x[inlier], y[inlier], w[inlier], cfg.eps)
    if not np.isfinite(scale) or scale <= cfg.eps:
        return 1.0, 0.0, 0.0
    final_residual = y[inlier] - (scale * x[inlier] + offset)
    target_range = float(np.quantile(y, 0.9) - np.quantile(y, 0.1)) + cfg.eps
    residual_score = np.exp(-float(np.median(np.abs(final_residual))) / (0.05 * target_range + cfg.eps))
    coverage = min(1.0, float(np.count_nonzero(inlier)) / max(cfg.min_fit_pixels * 2, 1))
    confidence = float(np.clip(residual_score * coverage * np.mean(np.clip(w[inlier], 0.0, 1.0)), 0.0, 1.0))
    return float(scale), float(offset), confidence


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = 0.5 * float(np.sum(ordered_weights))
    index = min(int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left")), len(values) - 1)
    return float(ordered_values[index])


def _isotonic_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Small weighted pool-adjacent-violators solver for monotonic LUT nodes."""
    block_values: list[float] = []
    block_weights: list[float] = []
    block_counts: list[int] = []
    for value, weight in zip(values.astype(np.float64), weights.astype(np.float64)):
        block_values.append(float(value))
        block_weights.append(max(float(weight), 1e-9))
        block_counts.append(1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_value = (
                block_values[-2] * block_weights[-2] + block_values[-1] * block_weights[-1]
            ) / merged_weight
            merged_count = block_counts[-2] + block_counts[-1]
            block_values[-2:] = [merged_value]
            block_weights[-2:] = [merged_weight]
            block_counts[-2:] = [merged_count]
    return np.concatenate(
        [np.full(count, value, np.float64) for value, count in zip(block_values, block_counts)]
    ).astype(np.float32)


def _apply_lut(values: np.ndarray, lut_x: np.ndarray, lut_y: np.ndarray, eps: float) -> np.ndarray:
    """Piecewise-linear LUT with linear endpoint extrapolation."""
    source = values.astype(np.float32)
    result = np.full(source.shape, np.nan, np.float32)
    finite = np.isfinite(source)
    if not np.any(finite):
        return result
    x = lut_x.astype(np.float32)
    y = lut_y.astype(np.float32)
    if len(x) < 2:
        result[finite] = source[finite]
        return result
    samples = source[finite]
    mapped = np.interp(samples, x, y).astype(np.float32)
    left_slope = max(float((y[1] - y[0]) / max(x[1] - x[0], eps)), eps)
    right_slope = max(float((y[-1] - y[-2]) / max(x[-1] - x[-2], eps)), eps)
    left = samples < x[0]
    right = samples > x[-1]
    mapped[left] = y[0] + left_slope * (samples[left] - x[0])
    mapped[right] = y[-1] + right_slope * (samples[right] - x[-1])
    result[finite] = mapped
    return result


def robust_monotonic_lut_samples(
    source: np.ndarray,
    target: np.ndarray,
    weights: Optional[np.ndarray],
    cfg: DepthSyncConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a bounded monotonic mapping from paired samples using robust bins."""
    good = np.isfinite(source) & np.isfinite(target)
    if weights is not None:
        good &= np.isfinite(weights) & (weights > 0)
    x = source[good].astype(np.float64)
    y = target[good].astype(np.float64)
    w = np.ones_like(x) if weights is None else weights[good].astype(np.float64)
    if x.size < cfg.min_fit_pixels or np.std(x) < cfg.eps:
        return np.array([0.0, 1.0], np.float32), np.array([0.0, 1.0], np.float32), 0.0

    # Do not pre-filter with an affine residual.  The purpose of this fit is to
    # represent strongly non-linear model-to-model mappings; an affine gate can
    # reject precisely the samples that justify a LUT (scene 12 regression).
    order = np.argsort(x)
    groups = [group for group in np.array_split(order, min(cfg.lut_nodes, len(order))) if len(group)]
    lut_x = np.array([_weighted_median(x[group], w[group]) for group in groups], np.float32)
    lut_y = np.array([_weighted_median(y[group], w[group]) for group in groups], np.float32)
    node_weights = np.array([float(np.sum(w[group])) for group in groups], np.float32)

    keep = np.concatenate(([True], np.diff(lut_x) > cfg.eps))
    lut_x, lut_y, node_weights = lut_x[keep], lut_y[keep], node_weights[keep]
    if len(lut_x) < 2:
        return np.array([0.0, 1.0], np.float32), np.array([0.0, 1.0], np.float32), 0.0
    lut_y = _isotonic_increasing(lut_y, node_weights)
    prediction = _apply_lut(x.astype(np.float32), lut_x, lut_y, cfg.eps)
    target_range = float(np.quantile(y, 0.9) - np.quantile(y, 0.1)) + cfg.eps
    residual_score = np.exp(-float(np.median(np.abs(y - prediction))) / (0.05 * target_range + cfg.eps))
    coverage = min(1.0, float(x.size) / max(cfg.min_fit_pixels * 2, 1))
    confidence = float(np.clip(residual_score * coverage * np.mean(np.clip(w, 0.0, 1.0)), 0.0, 1.0))
    return lut_x, lut_y, confidence


def _robust_value_mask(values: np.ndarray, cfg: DepthSyncConfig) -> np.ndarray:
    """Return fit-valid values, excluding a dominant invalid-floor plateau."""
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    samples = array[finite]
    if samples.size < cfg.min_fit_pixels:
        return finite
    low, high = np.quantile(
        samples, (cfg.lut_quantile_low, cfg.lut_quantile_high)
    )
    robust_range = max(float(high - low), cfg.eps)
    floor = float(np.quantile(samples, cfg.invalid_floor_quantile))
    tolerance = max(
        cfg.invalid_floor_tolerance_fraction * robust_range,
        16.0 * cfg.eps,
    )
    plateau = np.abs(samples - floor) <= tolerance
    above = samples > floor + tolerance
    if (
        float(np.mean(plateau)) >= cfg.invalid_floor_fraction
        and int(np.count_nonzero(above)) >= cfg.min_fit_pixels
    ):
        finite &= array > floor + tolerance
    return finite


def _rank_correlation(x: np.ndarray, y: np.ndarray, eps: float) -> float:
    """Deterministic Spearman-like correlation without a scipy dependency."""
    if x.size < 2 or np.std(x) <= eps or np.std(y) <= eps:
        return 0.0
    rx = np.empty(x.size, np.float64)
    ry = np.empty(y.size, np.float64)
    rx[np.argsort(x, kind="mergesort")] = np.arange(x.size, dtype=np.float64)
    ry[np.argsort(y, kind="mergesort")] = np.arange(y.size, dtype=np.float64)
    value = float(np.corrcoef(rx, ry)[0, 1])
    return value if np.isfinite(value) else 0.0


def quantile_monotonic_lut(
    source: np.ndarray,
    target: np.ndarray,
    nodes: int,
    cfg: DepthSyncConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a correspondence-free monotonic LUT from matched quantiles."""
    source_values = np.asarray(source, dtype=np.float32)
    target_values = np.asarray(target, dtype=np.float32)
    source_values = source_values[_robust_value_mask(source_values, cfg)]
    target_values = target_values[_robust_value_mask(target_values, cfg)]
    if min(source_values.size, target_values.size) < cfg.min_fit_pixels:
        return (
            np.array([0.0, 1.0], np.float32),
            np.array([0.0, 1.0], np.float32),
            0.0,
        )
    quantiles = np.linspace(
        cfg.lut_quantile_low,
        cfg.lut_quantile_high,
        max(int(nodes), 2),
        dtype=np.float64,
    )
    lut_x = np.quantile(source_values, quantiles).astype(np.float32)
    lut_y = np.quantile(target_values, quantiles).astype(np.float32)
    keep = np.concatenate(([True], np.diff(lut_x) > cfg.eps))
    lut_x, lut_y = lut_x[keep], lut_y[keep]
    if len(lut_x) < 2:
        return (
            np.array([0.0, 1.0], np.float32),
            np.array([0.0, 1.0], np.float32),
            0.0,
        )
    lut_y = _isotonic_increasing(lut_y, np.ones_like(lut_y))
    return lut_x, lut_y, 1.0


def _stabilized_frame_lut(
    frame: np.ndarray,
    anchor: np.ndarray,
    anchor_lut_x: np.ndarray,
    anchor_lut_y: np.ndarray,
    cfg: DepthSyncConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compose frame→anchor quantiles with the anchor→photo mapping."""
    frame_x, anchor_values, confidence = quantile_monotonic_lut(
        frame, anchor, min(cfg.lut_temporal_nodes, cfg.lut_nodes), cfg
    )
    if confidence <= 0 or len(frame_x) < 2:
        return anchor_lut_x.copy(), anchor_lut_y.copy(), 0.0
    mapped_y = _apply_lut(
        anchor_values, anchor_lut_x, anchor_lut_y, cfg.eps
    )
    base_y = _apply_lut(frame_x, anchor_lut_x, anchor_lut_y, cfg.eps)
    anchor_target = _apply_lut(anchor, anchor_lut_x, anchor_lut_y, cfg.eps)
    valid_target = anchor_target[np.isfinite(anchor_target)]
    target_range = (
        float(np.quantile(valid_target, 0.9) - np.quantile(valid_target, 0.1))
        if valid_target.size
        else 1.0
    )
    correction_limit = cfg.lut_distribution_clip_fraction * max(
        target_range, cfg.eps
    )
    correction = np.clip(mapped_y - base_y, -correction_limit, correction_limit)
    blended_y = base_y + np.clip(cfg.lut_distribution_blend, 0.0, 1.0) * correction
    return (*_canonical_lut(frame_x, blended_y, cfg.lut_nodes, cfg.eps), confidence)


def _mapping_score(
    source_image: np.ndarray,
    target_image: np.ndarray,
    lut_x: np.ndarray,
    lut_y: np.ndarray,
    cfg: DepthSyncConfig,
    paired: bool,
) -> float:
    """Score paired accuracy and independent output-distribution alignment."""
    source_valid = _robust_value_mask(source_image, cfg)
    target_valid = _robust_value_mask(target_image, cfg)
    if min(int(np.count_nonzero(source_valid)), int(np.count_nonzero(target_valid))) < cfg.min_fit_pixels:
        return float("inf")
    prediction = _apply_lut(source_image, lut_x, lut_y, cfg.eps)
    target_values = target_image[target_valid]
    target_range = max(
        float(np.quantile(target_values, 0.9) - np.quantile(target_values, 0.1)),
        cfg.eps,
    )
    quantiles = np.asarray([0.1, 0.25, 0.5, 0.75, 0.9])
    distribution_error = float(
        np.mean(
            np.abs(
                np.quantile(prediction[source_valid], quantiles)
                - np.quantile(target_values, quantiles)
            )
        )
        / target_range
    )
    paired_error = 0.0
    if paired:
        valid_pair = source_valid & target_valid & np.isfinite(prediction)
        if int(np.count_nonzero(valid_pair)) < cfg.min_fit_pixels:
            return float("inf")
        paired_error = float(
            np.median(np.abs(prediction[valid_pair] - target_image[valid_pair]))
            / target_range
        )
    return paired_error + cfg.lut_distribution_weight * distribution_error


def _sequence_distribution_jump(
    frames: Sequence[np.ndarray], cfg: DepthSyncConfig
) -> float:
    """P95 adjacent-frame quantile jump used to detect genuine model flicker."""
    quantiles = np.linspace(0.1, 0.9, 9)
    curves: list[np.ndarray] = []
    for frame in frames:
        values = np.asarray(frame, np.float32)
        values = values[_robust_value_mask(values, cfg)]
        if values.size < cfg.min_fit_pixels:
            continue
        curve = np.quantile(values, quantiles)
        scale = max(float(curve[-1] - curve[0]), cfg.eps)
        curves.append(((curve - curve[0]) / scale).astype(np.float32))
    if len(curves) < 2:
        return 0.0
    jumps = np.mean(np.abs(np.diff(np.asarray(curves), axis=0)), axis=1)
    return float(np.quantile(jumps, 0.95))


def _adjacent_distribution_jump(
    first: np.ndarray, second: np.ndarray, cfg: DepthSyncConfig
) -> float:
    """Quantile-shape distance invariant to global depth scale and shift."""
    curves: list[np.ndarray] = []
    quantiles = np.linspace(0.1, 0.9, 9)
    for frame in (first, second):
        values = np.asarray(frame, np.float32)
        values = values[_robust_value_mask(values, cfg)]
        if values.size < cfg.min_fit_pixels:
            return float("inf")
        curve = np.quantile(values, quantiles)
        scale = max(float(curve[-1] - curve[0]), cfg.eps)
        curves.append((curve - curve[0]) / scale)
    return float(np.mean(np.abs(curves[1] - curves[0])))


def _bilinear_sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    map_x = xs.astype(np.float32).reshape(-1, 1)
    map_y = ys.astype(np.float32).reshape(-1, 1)
    return cv2.remap(image.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan).reshape(-1)


def _sample_grid_field(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    gh, gw = field.shape[:2]
    gx = xs * max(gw - 1, 1) / max(shape[1] - 1, 1)
    gy = ys * max(gh - 1, 1) / max(shape[0] - 1, 1)
    sampled = [_bilinear_sample(field[..., c], gx, gy) for c in range(field.shape[2] if field.ndim == 3 else 1)]
    return np.stack(sampled, axis=1) if field.ndim == 3 else sampled[0]


def _deterministic_pick(indices: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or indices.size == 0:
        return np.empty(0, np.int64)
    if indices.size <= count:
        return indices
    return indices[np.linspace(0, indices.size - 1, count, dtype=np.int64)]


def _sample_coordinates(depth: np.ndarray, cfg: DepthSyncConfig, face_box: Optional[tuple[float, float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    valid = _valid(depth)
    filled = np.nan_to_num(depth, nan=float(np.nanmedian(depth[valid])) if np.any(valid) else 0.0)
    gx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    if np.any(valid):
        valid &= gradient <= np.quantile(gradient[valid], cfg.edge_quantile)
    flat = np.flatnonzero(valid)
    if face_box is None:
        chosen = _deterministic_pick(flat, cfg.sample_count)
    else:
        h, w = depth.shape
        x0, y0, x1, y1 = face_box
        yy, xx = np.mgrid[:h, :w]
        face = (xx >= x0 * w) & (xx < x1 * w) & (yy >= y0 * h) & (yy < y1 * h)
        face_indices = np.flatnonzero(valid & face)
        other_indices = np.flatnonzero(valid & ~face)
        face_count = int(round(cfg.sample_count * cfg.face_sample_fraction))
        chosen = np.concatenate(
            (_deterministic_pick(face_indices, face_count), _deterministic_pick(other_indices, cfg.sample_count - face_count))
        )
    ys, xs = np.unravel_index(chosen, depth.shape)
    return xs.astype(np.float32), ys.astype(np.float32)


def _align_photo(photo: np.ndarray, shape: tuple[int, int], photo_to_video_grid: Optional[np.ndarray]) -> np.ndarray:
    if photo_to_video_grid is None:
        return _resize(photo, shape, cv2.INTER_AREA)
    grid = photo_to_video_grid.astype(np.float32)
    if grid.shape[:2] != shape:
        grid = cv2.resize(grid, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    map_x = grid[..., 0] * max(photo.shape[1] - 1, 1)
    map_y = grid[..., 1] * max(photo.shape[0] - 1, 1)
    return cv2.remap(photo, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


def _canonical_lut(lut_x: np.ndarray, lut_y: np.ndarray, nodes: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    if len(lut_x) < 2 or float(lut_x[-1] - lut_x[0]) <= eps:
        x = np.linspace(0.0, 1.0, nodes, dtype=np.float32)
        return x, x.copy()
    x = np.linspace(float(lut_x[0]), float(lut_x[-1]), nodes, dtype=np.float32)
    y = _apply_lut(x, lut_x, lut_y, eps)
    return x, y


def _affine_lut(source: np.ndarray, scale: float, offset: float, cfg: DepthSyncConfig) -> tuple[np.ndarray, np.ndarray]:
    finite = source[np.isfinite(source)]
    low, high = np.quantile(finite, (0.02, 0.98)) if finite.size else (0.0, 1.0)
    if high <= low + cfg.eps:
        high = low + 1.0
    x = np.linspace(float(low), float(high), cfg.lut_nodes, dtype=np.float32)
    return x, (scale * x + offset).astype(np.float32)


class DepthSync:
    """Estimate the compact V4.1 global monotonic LUT sequence."""

    def __init__(self, config: Optional[DepthSyncConfig] = None):
        self.cfg = config or DepthSyncConfig()
        if self.cfg.lut_nodes < 2:
            raise ValueError("lut_nodes must be at least 2")

    def offline_prepare(
        self,
        video_depths: Sequence[np.ndarray],
        photo_depth: np.ndarray,
        anchor_index: int,
        motion: Optional[MotionSequence] = None,
        face_box: Optional[tuple[float, float, float, float]] = None,
        photo_to_video_grid: Optional[np.ndarray] = None,
    ) -> SyncResult:
        if len(video_depths) == 0:
            raise ValueError("video_depths is empty")
        if not 0 <= anchor_index < len(video_depths):
            raise ValueError("anchor_index is out of range")
        if motion is not None and (len(motion.to_previous) != len(video_depths) or len(motion.to_next) != len(video_depths)):
            raise ValueError("motion fields and video_depths must have equal length")
        cfg, n = self.cfg, len(video_depths)
        shape = video_depths[anchor_index].shape[:2]
        raw = [_resize(_to_working(d, cfg.depth_mode, cfg.eps), shape) for d in video_depths]
        photo = _align_photo(_to_working(photo_depth, cfg.depth_mode, cfg.eps), shape, photo_to_video_grid)
        valid_photo = photo[np.isfinite(photo)]
        photo_range = float(
            np.quantile(valid_photo, 0.9) - np.quantile(valid_photo, 0.1)
        ) if valid_photo.size else 1.0
        base_outputs: list[Optional[np.ndarray]] = [None] * n
        scales, offsets, confidences = np.ones(n), np.zeros(n), np.zeros(n)
        reasons = [""] * n
        lut_x = np.zeros((n, cfg.lut_nodes), np.float32)
        lut_y = np.zeros_like(lut_x)

        xs, ys = _sample_coordinates(raw[anchor_index], cfg, face_box)
        anchor_source = _bilinear_sample(raw[anchor_index], xs, ys)
        anchor_target = _bilinear_sample(photo, xs, ys)
        a, b, affine_confidence = robust_affine_samples(anchor_source, anchor_target, None, cfg)
        anchor_lut_x, anchor_lut_y = _affine_lut(anchor_source, a, b, cfg)
        confidence = affine_confidence
        pair_correlation = 0.0
        best_score = _mapping_score(
            raw[anchor_index],
            photo,
            anchor_lut_x,
            anchor_lut_y,
            cfg,
            paired=True,
        )
        source_fit_valid = _robust_value_mask(anchor_source, cfg)
        target_fit_valid = _robust_value_mask(anchor_target, cfg)
        paired_fit_valid = source_fit_valid & target_fit_valid
        pair_correlation = _rank_correlation(
            anchor_source[paired_fit_valid],
            anchor_target[paired_fit_valid],
            cfg.eps,
        )
        candidate_counts = tuple(
            sorted(
                {
                    int(nodes)
                    for nodes in cfg.lut_candidate_nodes
                    if 2 <= int(nodes) <= cfg.lut_nodes
                }
            )
        ) or (cfg.lut_nodes,)
        if pair_correlation >= cfg.lut_min_pair_correlation:
            paired_source = anchor_source[paired_fit_valid]
            paired_target = anchor_target[paired_fit_valid]
            for candidate_nodes in candidate_counts:
                candidate_cfg = replace(cfg, lut_nodes=candidate_nodes)
                candidate_x, candidate_y, lut_confidence = (
                    robust_monotonic_lut_samples(
                        paired_source,
                        paired_target,
                        None,
                        candidate_cfg,
                    )
                )
                if lut_confidence <= 0:
                    continue
                canonical_x, canonical_y = _canonical_lut(
                    candidate_x,
                    candidate_y,
                    cfg.lut_nodes,
                    cfg.eps,
                )
                score = _mapping_score(
                    raw[anchor_index],
                    photo,
                    canonical_x,
                    canonical_y,
                    cfg,
                    paired=True,
                ) + cfg.lut_complexity_penalty * (
                    candidate_nodes / max(cfg.lut_nodes, 1)
                )
                if score < best_score:
                    best_score = score
                    anchor_lut_x, anchor_lut_y = canonical_x, canonical_y
                    confidence = lut_confidence

        # Correspondence-free distribution mapping is the mandatory
        # fallback when aligned pixels are contradictory or insufficient.
        # It guarantees a globally useful value domain for revealed areas
        # instead of silently returning an identity LUT.
        raw_score = _mapping_score(
            raw[anchor_index],
            photo,
            np.asarray([0.0, 1.0], np.float32),
            np.asarray([0.0, 1.0], np.float32),
            cfg,
            paired=False,
        )
        if not np.isfinite(best_score) or pair_correlation < cfg.lut_min_pair_correlation:
            quantile_best = float("inf")
            quantile_best_x: np.ndarray | None = None
            quantile_best_y: np.ndarray | None = None
            quantile_best_confidence = 0.0
            for candidate_nodes in candidate_counts:
                candidate_x, candidate_y, quantile_confidence = (
                    quantile_monotonic_lut(
                        raw[anchor_index], photo, candidate_nodes, cfg
                    )
                )
                if quantile_confidence <= 0:
                    continue
                canonical_x, canonical_y = _canonical_lut(
                    candidate_x,
                    candidate_y,
                    cfg.lut_nodes,
                    cfg.eps,
                )
                score = _mapping_score(
                    raw[anchor_index],
                    photo,
                    canonical_x,
                    canonical_y,
                    cfg,
                    paired=False,
                ) + cfg.lut_complexity_penalty * (
                    candidate_nodes / max(cfg.lut_nodes, 1)
                )
                if score < quantile_best:
                    quantile_best = score
                    quantile_best_x, quantile_best_y = canonical_x, canonical_y
                    quantile_best_confidence = quantile_confidence
            if np.isfinite(quantile_best) and quantile_best < raw_score:
                best_score = quantile_best
                assert quantile_best_x is not None and quantile_best_y is not None
                anchor_lut_x, anchor_lut_y = quantile_best_x, quantile_best_y
                confidence = quantile_best_confidence
                reasons[anchor_index] = "anchor_quantile_fallback"
        if confidence <= 0:
            reasons[anchor_index] = "insufficient_anchor_support"
        scales[anchor_index], offsets[anchor_index] = 1.0, 0.0
        confidences[anchor_index] = confidence
        lut_x[anchor_index], lut_y[anchor_index] = anchor_lut_x, anchor_lut_y
        base_outputs[anchor_index] = _apply_lut(raw[anchor_index], anchor_lut_x, anchor_lut_y, cfg.eps)
        sequence_distribution_jump = (
            _sequence_distribution_jump(raw, cfg)
            if cfg.enable_lut_distribution_stabilization
            else 0.0
        )
        strong_distribution_flicker = (
            sequence_distribution_jump >= cfg.lut_frame_force_jump_threshold
        )
        stabilize_frame_distribution = (
            cfg.enable_lut_distribution_stabilization
            and (
                pair_correlation >= cfg.lut_frame_min_pair_correlation
                or strong_distribution_flicker
            )
        )
        frame_lut_cfg = (
            replace(
                cfg,
                lut_temporal_nodes=cfg.lut_nodes,
                lut_distribution_clip_fraction=1.0e6,
            )
            if strong_distribution_flicker
            else cfg
        )
        for direction in (-1, 1):
            previous_index = anchor_index
            for i in range(anchor_index + direction, -1 if direction < 0 else n, direction):
                if (
                    cfg.enable_lut_distribution_stabilization
                    and stabilize_frame_distribution
                ):
                    if (
                        strong_distribution_flicker
                        and _adjacent_distribution_jump(
                            raw[i], raw[previous_index], cfg
                        )
                        >= cfg.lut_frame_freeze_jump_threshold
                    ):
                        scales[i], offsets[i] = 1.0, 0.0
                        confidences[i] = 0.0
                        lut_x[i] = lut_x[previous_index]
                        lut_y[i] = lut_y[previous_index]
                        base_outputs[i] = _apply_lut(
                            raw[i], lut_x[i], lut_y[i], cfg.eps
                        )
                        reasons[i] = "distribution_change_freeze"
                        previous_index = i
                        continue
                    current_x, current_y, frame_confidence = _stabilized_frame_lut(
                        raw[i],
                        raw[anchor_index],
                        anchor_lut_x,
                        anchor_lut_y,
                        frame_lut_cfg,
                    )
                    # Keep the photo anchor exact and unlock distribution
                    # compensation gradually.  This is computed offline and
                    # prevents an anchor→next-frame mapping discontinuity.
                    distance = abs(i - anchor_index)
                    transition = float(
                        np.clip(
                            distance / max(cfg.anchor_transition_radius, 1),
                            0.0,
                            1.0,
                        )
                    )
                    anchor_y_at_current_x = _apply_lut(
                        current_x, anchor_lut_x, anchor_lut_y, cfg.eps
                    )
                    current_y = (
                        (1.0 - transition) * anchor_y_at_current_x
                        + transition * current_y
                    ).astype(np.float32)
                    scales[i], offsets[i] = 1.0, 0.0
                    confidences[i] = frame_confidence
                    lut_x[i], lut_y[i] = current_x, current_y
                    base_outputs[i] = _apply_lut(
                        raw[i], current_x, current_y, cfg.eps
                    )
                    if frame_confidence <= 0:
                        reasons[i] = "insufficient_distribution_support"
                    previous_index = i
                    continue
                if not cfg.enable_lut_temporal_adjustment:
                    scales[i], offsets[i] = 1.0, 0.0
                    confidences[i] = confidence
                    lut_x[i], lut_y[i] = anchor_lut_x, anchor_lut_y
                    base_outputs[i] = _apply_lut(
                        raw[i], anchor_lut_x, anchor_lut_y, cfg.eps
                    )
                    previous_index = i
                    continue
                # Global mapping propagates against the previous base map. The
                # spatial photo residual travels separately to avoid double use.
                previous = base_outputs[previous_index]
                assert previous is not None
                xs, ys = _sample_coordinates(raw[i], cfg, face_box)
                pre_mapped = _apply_lut(raw[i], anchor_lut_x, anchor_lut_y, cfg.eps)
                source = _bilinear_sample(pre_mapped, xs, ys)
                weights = np.ones_like(source, np.float32)
                mapped_x, mapped_y = xs.copy(), ys.copy()
                if motion is not None:
                    if direction > 0:
                        field = motion.to_previous[i]
                        conf_field = None if motion.confidence_previous is None else motion.confidence_previous[i]
                    else:
                        field = motion.to_next[i]
                        conf_field = None if motion.confidence_next is None else motion.confidence_next[i]
                    displacement = _sample_grid_field(field, xs, ys, shape)
                    mapped_x += displacement[:, 0] * max(shape[1] - 1, 1)
                    mapped_y += displacement[:, 1] * max(shape[0] - 1, 1)
                    if conf_field is not None:
                        weights *= np.clip(_sample_grid_field(conf_field[..., None], xs, ys, shape)[:, 0], 0.0, 1.0)
                target = _bilinear_sample(previous, mapped_x, mapped_y)
                in_bounds = (mapped_x >= 0) & (mapped_x < shape[1] - 1) & (mapped_y >= 0) & (mapped_y < shape[0] - 1)
                weights *= in_bounds.astype(np.float32)
                candidate_a, candidate_b, fit_conf = robust_affine_samples(source, target, weights, cfg)

                prev_a, prev_b = scales[previous_index], offsets[previous_index]
                if fit_conf <= 0:
                    a, b = prev_a, prev_b
                    current_x, current_y = lut_x[previous_index], lut_y[previous_index]
                    reasons[i] = "insufficient_temporal_support"
                else:
                    alpha = float(np.clip(cfg.parameter_smoothing * fit_conf, 0.0, 1.0))
                    distance = abs(i - anchor_index)
                    transition_span = max(cfg.anchor_transition_radius - cfg.anchor_lock_radius, 1)
                    transition = np.clip(
                        (distance - cfg.anchor_lock_radius) / transition_span,
                        0.0,
                        1.0,
                    )
                    alpha *= float(transition)
                    a = (1.0 - alpha) * prev_a + alpha * candidate_a
                    b = (1.0 - alpha) * prev_b + alpha * candidate_b
                    scale_delta = cfg.lut_max_scale_delta
                    offset_fraction = cfg.lut_max_offset_delta_fraction
                    a = float(np.clip(a, prev_a * (1.0 - scale_delta), prev_a * (1.0 + scale_delta)))
                    max_offset_delta = offset_fraction * max(photo_range, cfg.eps)
                    b = float(np.clip(b, prev_b - max_offset_delta, prev_b + max_offset_delta))
                    current_x = anchor_lut_x
                    current_y = (a * anchor_lut_y + b).astype(np.float32)
                if abs(i - anchor_index) <= cfg.anchor_lock_radius:
                    a, b = scales[anchor_index], offsets[anchor_index]
                    current_x, current_y = lut_x[anchor_index], lut_y[anchor_index]
                    reasons[i] = "anchor_lock"
                scales[i], offsets[i], confidences[i] = a, b, fit_conf
                lut_x[i], lut_y[i] = current_x, current_y
                base_outputs[i] = _apply_lut(raw[i], current_x, current_y, cfg.eps)
                previous_index = i

        if (
            face_box is not None
            and cfg.subject_offset_clip_fraction > 0
        ):
            x0, y0, x1, y1 = face_box
            subject_ys = slice(
                max(int(y0 * shape[0]), 0),
                min(int(np.ceil(y1 * shape[0])), shape[0]),
            )
            subject_xs = slice(
                max(int(x0 * shape[1]), 0),
                min(int(np.ceil(x1 * shape[1])), shape[1]),
            )
            anchor_base = base_outputs[anchor_index]
            assert anchor_base is not None
            subject_delta = photo[subject_ys, subject_xs] - anchor_base[
                subject_ys, subject_xs
            ]
            finite_delta = subject_delta[np.isfinite(subject_delta)]
            if finite_delta.size:
                limit = cfg.subject_offset_clip_fraction * max(photo_range, cfg.eps)
                delta = float(np.clip(np.median(finite_delta), -limit, limit))
                lut_y += delta
                offsets += delta
                base_outputs = [
                    None if base is None else (base + delta).astype(np.float32)
                    for base in base_outputs
                ]

        concrete_base_outputs = [base for base in base_outputs if base is not None]
        assert len(concrete_base_outputs) == n
        depths = np.stack(
            [_from_working(x, cfg.depth_mode, cfg.eps) for x in concrete_base_outputs]
        ).astype(np.float32)
        return SyncResult(
            depths,
            scales.astype(np.float32),
            offsets.astype(np.float32),
            confidences.astype(np.float32),
            tuple(reasons),
            lut_x,
            lut_y,
        )

    def apply_frame(
        self,
        depth: np.ndarray,
        params: FrameParameters,
        output_shape: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        working = _to_working(depth, self.cfg.depth_mode, self.cfg.eps)
        if params.lut_x is None or params.lut_y is None:
            raise ValueError("V4.1 frame parameters require lut_x and lut_y")
        synced = _apply_lut(working, params.lut_x, params.lut_y, self.cfg.eps)
        if output_shape is not None and synced.shape[:2] != output_shape:
            synced = _resize(synced, output_shape, self.cfg.high_res_interpolation)
        return _from_working(synced, self.cfg.depth_mode, self.cfg.eps).astype(np.float32)

    def __call__(
        self,
        video_depths: Sequence[np.ndarray],
        photo_depth: np.ndarray,
        anchor_index: int,
    ) -> SyncResult:
        return self.offline_prepare(
            video_depths,
            photo_depth,
            anchor_index,
        )
