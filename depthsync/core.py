"""Parameter-only photo-anchor depth synchronization for edge deployment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthSyncConfig:
    """Configuration for the lightweight parameter-estimation path."""

    algorithm_version: str = "v4"
    depth_mode: str = "disparity"
    sample_count: int = 2048
    face_sample_fraction: float = 0.30
    edge_quantile: float = 0.70
    mad_threshold: float = 3.0
    parameter_smoothing: float = 0.65
    max_scale_delta: float = 0.03
    max_offset_delta_fraction: float = 0.02
    anchor_lock_radius: int = 1
    min_fit_pixels: int = 128
    high_res_interpolation: int = cv2.INTER_LINEAR
    mapping_mode: str = "lut"
    lut_nodes: int = 8
    residual_grid_shape: tuple[int, int] = (9, 16)
    static_grid_shape: tuple[int, int] = (72, 128)
    static_target_shape: tuple[int, int] = (72, 128)
    residual_radius: int = 30
    residual_clip_fraction: float = 0.35
    residual_blur_sigma: float = 0.8
    anchor_transition_radius: int = 6
    lut_max_scale_delta: float = 0.01
    lut_max_offset_delta_fraction: float = 0.005
    residual_motion_strength: float = 1.0
    static_motion_threshold: float = 0.0025
    static_motion_softness: float = 0.0005
    static_depth_threshold_fraction: float = 0.10
    static_depth_softness_fraction: float = 0.02
    static_depth_range_fraction: float = 0.35
    static_depth_range_softness_fraction: float = 0.05
    static_temporal_median_radius: int = 1
    static_occupancy_barrier_threshold: float = 0.50
    static_occupancy_barrier_radius: int = 2
    static_subject_box_margin: int = 6
    static_region_link_threshold_fraction: float = 0.05
    subject_offset_clip_fraction: float = 0.10
    static_foreground_quantile: float = 0.85
    static_foreground_softness_fraction: float = 0.05
    static_face_expand_x: float = 0.20
    static_face_expand_top: float = 0.20
    static_face_expand_bottom: float = 0.50
    static_edge_threshold_fraction: float = 0.0
    static_edge_softness_fraction: float = 0.15
    static_erode_radius: int = 0
    static_close_radius: int = 2
    static_mask_blur_sigma: float = 1.0
    static_expand_radius: int = 2
    correction_edge_kernel: int = 7
    static_correction_edge_kernel: int = 1
    correction_edge_threshold_fraction: float = 0.50
    correction_edge_softness_fraction: float = 0.10
    static_min_confidence: float = 0.30
    region_grid_shape: tuple[int, int] = (72, 128)
    region_link_threshold_fraction: float = 0.04
    region_min_area_fraction: float = 0.01
    region_max_count: int = 8
    region_static_support: float = 0.65
    region_offset_clip_fraction: float = 0.75
    region_offset_step_fraction: float = 0.75
    region_offset_smoothing: float = 1.0
    region_shape_open_radius: int = 2
    region_edge_kernel: int = 1
    region_motion_min_confidence: float = 0.30
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
    residual_grid: Optional[np.ndarray] = None
    static_mask: Optional[np.ndarray] = None
    static_target_grid: Optional[np.ndarray] = None
    guidance_range: float = 0.0
    region_labels: Optional[np.ndarray] = None
    region_scales: Optional[np.ndarray] = None
    region_offsets: Optional[np.ndarray] = None
    region_shift: Optional[np.ndarray] = None


@dataclass
class SyncResult:
    depths: np.ndarray
    scales: np.ndarray
    offsets: np.ndarray
    confidences: np.ndarray
    fallback_reasons: tuple[str, ...]
    lut_x: np.ndarray
    lut_y: np.ndarray
    residual_grids: np.ndarray
    static_mask: np.ndarray
    static_target_grid: np.ndarray
    guidance_range: float
    region_labels: np.ndarray
    region_scales: np.ndarray
    region_offsets: np.ndarray
    region_shifts: np.ndarray

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
                residual,
                self.static_mask,
                self.static_target_grid,
                self.guidance_range,
                self.region_labels,
                self.region_scales,
                region_offset,
                region_shift,
            )
            for a, b, c, reason, x, y, residual, region_offset, region_shift in zip(
                self.scales,
                self.offsets,
                self.confidences,
                self.fallback_reasons,
                self.lut_x,
                self.lut_y,
                self.residual_grids,
                self.region_offsets,
                self.region_shifts,
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


def robust_affine(source: np.ndarray, target: np.ndarray, mask: np.ndarray, min_pixels: int) -> tuple[float, float, float]:
    """Compatibility wrapper for the original public helper."""
    cfg = DepthSyncConfig(min_fit_pixels=min_pixels)
    return robust_affine_samples(source[mask], target[mask], None, cfg)


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

    # Remove gross mismatches using the same robust affine initializer as V1.
    affine_a, affine_b, _ = robust_affine_samples(x, y, w, cfg)
    residual = y - (affine_a * x + affine_b)
    center = float(np.median(residual))
    sigma = 1.4826 * float(np.median(np.abs(residual - center))) + cfg.eps
    inlier = np.abs(residual - center) <= cfg.mad_threshold * sigma
    x, y, w = x[inlier], y[inlier], w[inlier]
    if x.size < cfg.min_fit_pixels:
        lo, hi = np.quantile(source[good], (0.1, 0.9))
        lut_x = np.array([lo, hi], np.float32)
        return lut_x, (affine_a * lut_x + affine_b).astype(np.float32), 0.0

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


def _make_residual_grid(residual: np.ndarray, photo_range: float, cfg: DepthSyncConfig) -> np.ndarray:
    grid_h, grid_w = cfg.residual_grid_shape
    if grid_h <= 0 or grid_w <= 0 or cfg.residual_radius <= 0:
        return np.zeros((0, 0), np.float32)
    finite = np.isfinite(residual)
    if not np.any(finite):
        return np.zeros((grid_h, grid_w), np.float32)
    fill = float(np.median(residual[finite]))
    cleaned = np.nan_to_num(residual, nan=fill, posinf=fill, neginf=fill).astype(np.float32)
    limit = cfg.residual_clip_fraction * max(photo_range, cfg.eps)
    cleaned = np.clip(cleaned, -limit, limit)
    if cfg.residual_blur_sigma > 0:
        cleaned = cv2.GaussianBlur(cleaned, (0, 0), cfg.residual_blur_sigma)
    return cv2.resize(cleaned, (grid_w, grid_h), interpolation=cv2.INTER_AREA).astype(np.float32)


def _warp_residual_grid(
    grid: np.ndarray,
    field: Optional[np.ndarray],
    motion_strength: float = 1.0,
) -> np.ndarray:
    if grid.size == 0 or field is None:
        return grid.copy()
    grid_h, grid_w = grid.shape
    dense = cv2.resize(field.astype(np.float32), (grid_w, grid_h), interpolation=cv2.INTER_LINEAR)
    yy, xx = np.mgrid[:grid_h, :grid_w].astype(np.float32)
    map_x = xx + motion_strength * dense[..., 0] * max(grid_w - 1, 1)
    map_y = yy + motion_strength * dense[..., 1] * max(grid_h - 1, 1)
    return cv2.remap(grid, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)


def _static_guidance_mask(
    motion: Optional[MotionSequence],
    base_outputs: Sequence[np.ndarray],
    photo: np.ndarray,
    photo_range: float,
    grid_shape: tuple[int, int],
    cfg: DepthSyncConfig,
    face_box: Optional[tuple[float, float, float, float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate cells that are stable in both motion and relative depth."""
    grid_h, grid_w = grid_shape
    if motion is None or grid_h <= 0 or grid_w <= 0:
        empty = np.zeros((max(grid_h, 0), max(grid_w, 0)), np.float32)
        return empty, empty.copy()
    magnitudes: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    for index in range(1, len(motion.to_previous)):
        field = motion.to_previous[index].astype(np.float32)
        magnitude = np.linalg.norm(field, axis=-1)
        magnitudes.append(cv2.resize(magnitude, (grid_w, grid_h), interpolation=cv2.INTER_AREA))
        if motion.confidence_previous is not None:
            confidences.append(
                cv2.resize(
                    motion.confidence_previous[index].astype(np.float32),
                    (grid_w, grid_h),
                    interpolation=cv2.INTER_AREA,
                )
            )
    motion_p90 = np.quantile(np.stack(magnitudes), 0.90, axis=0)
    stable = np.clip(
        (cfg.static_motion_threshold - motion_p90) / max(cfg.static_motion_softness, cfg.eps),
        0.0,
        1.0,
    ).astype(np.float32)
    motion_gate = stable.copy()

    # Remove each frame's global median before measuring temporal depth change.
    # This keeps genuinely static background eligible even when the lightweight
    # depth model's global scale drifts, while rejecting independently moving
    # subjects whose local depth changes relative to the scene.
    low_depths = np.stack(
        [cv2.resize(depth, (grid_w, grid_h), interpolation=cv2.INTER_AREA) for depth in base_outputs]
    )
    centered = low_depths - np.nanmedian(low_depths, axis=(1, 2), keepdims=True)
    temporal_center = np.nanmedian(centered, axis=0)
    temporal_mad = np.nanmedian(np.abs(centered - temporal_center), axis=0)
    depth_threshold = cfg.static_depth_threshold_fraction * max(photo_range, cfg.eps)
    depth_softness = cfg.static_depth_softness_fraction * max(photo_range, cfg.eps)
    depth_gate = np.clip(
        (depth_threshold - temporal_mad) / max(depth_softness, cfg.eps),
        0.0,
        1.0,
    ).astype(np.float32)
    stable *= depth_gate

    # Median MAD misses an object that occupies a background cell only near one
    # end of the clip. A short temporal median rejects isolated one-frame noise;
    # the following full range still catches a silhouette present for >=2 frames.
    temporal_filtered = centered
    temporal_radius = max(int(cfg.static_temporal_median_radius), 0)
    if temporal_radius > 0 and centered.shape[0] > 1:
        temporal_filtered = np.empty_like(centered)
        for index in range(centered.shape[0]):
            lo = max(index - temporal_radius, 0)
            hi = min(index + temporal_radius + 1, centered.shape[0])
            temporal_filtered[index] = np.nanmedian(centered[lo:hi], axis=0)
    temporal_range = np.nanmax(temporal_filtered, axis=0) - np.nanmin(
        temporal_filtered, axis=0
    )
    range_threshold = cfg.static_depth_range_fraction * max(photo_range, cfg.eps)
    range_softness = cfg.static_depth_range_softness_fraction * max(photo_range, cfg.eps)
    occupancy_gate = np.clip(
        (range_threshold - temporal_range) / max(range_softness, cfg.eps),
        0.0,
        1.0,
    ).astype(np.float32)
    stable *= occupancy_gate

    # Protect likely foreground only inside an expanded face ROI. Applying a
    # generic depth-edge gate everywhere created a wide fallback band at the
    # wall boundary, because the fixed photo layer was removed on both sides.
    photo_grid = cv2.resize(photo, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    finite_photo = photo_grid[np.isfinite(photo_grid)]
    foreground_gate = np.ones((grid_h, grid_w), np.float32)
    if finite_photo.size and face_box is not None and 0.0 < cfg.static_foreground_quantile < 1.0:
        foreground_threshold = float(np.quantile(finite_photo, cfg.static_foreground_quantile))
        foreground_softness = cfg.static_foreground_softness_fraction * max(photo_range, cfg.eps)
        candidate_gate = np.clip(
            (foreground_threshold + foreground_softness - photo_grid)
            / max(foreground_softness, cfg.eps),
            0.0,
            1.0,
        ).astype(np.float32)
        x0, y0, x1, y1 = face_box
        x0 = max(x0 - cfg.static_face_expand_x, 0.0)
        x1 = min(x1 + cfg.static_face_expand_x, 1.0)
        y0 = max(y0 - cfg.static_face_expand_top, 0.0)
        y1 = min(y1 + cfg.static_face_expand_bottom, 1.0)
        xs = slice(int(x0 * grid_w), max(int(np.ceil(x1 * grid_w)), 1))
        ys = slice(int(y0 * grid_h), max(int(np.ceil(y1 * grid_h)), 1))
        foreground_gate[ys, xs] = candidate_gate[ys, xs]

    edge_gate = np.ones((grid_h, grid_w), np.float32)
    if cfg.static_edge_threshold_fraction > 0:
        gradient = np.hypot(
            cv2.Sobel(photo_grid, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(photo_grid, cv2.CV_32F, 0, 1, ksize=3),
        )
        edge_threshold = cfg.static_edge_threshold_fraction * max(photo_range, cfg.eps)
        edge_softness = cfg.static_edge_softness_fraction * max(photo_range, cfg.eps)
        edge_gate = np.clip(
            (edge_threshold - gradient) / max(edge_softness, cfg.eps),
            0.0,
            1.0,
        ).astype(np.float32)
    if confidences:
        confidence = np.median(np.stack(confidences), axis=0)
        stable *= (confidence >= cfg.static_min_confidence).astype(np.float32)
    # Fill isolated block-MV/confidence holes and low-pass the weight field.
    # Occupancy and foreground gates are re-applied afterwards, so smoothing a
    # flat wall cannot leak the fixed photo layer back onto moving silhouettes.
    if cfg.static_close_radius > 0:
        radius = cfg.static_close_radius
        stable = cv2.morphologyEx(
            stable,
            cv2.MORPH_CLOSE,
            np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
        )
    if cfg.static_mask_blur_sigma > 0:
        stable = cv2.GaussianBlur(stable, (0, 0), cfg.static_mask_blur_sigma)
    if cfg.static_erode_radius > 0:
        radius = cfg.static_erode_radius
        edge_gate = cv2.erode(edge_gate, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    static_mask = np.clip(
        stable * occupancy_gate * foreground_gate * edge_gate,
        0.0,
        1.0,
    ).astype(np.float32)
    if cfg.static_expand_radius > 0:
        radius = cfg.static_expand_radius
        static_mask = cv2.dilate(
            static_mask,
            np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
        )
    # A silhouette-shaped all-clip exclusion leaves a visible "ghost contour"
    # in the background whenever the actor moves away. Convert the connected
    # moving subject track into one compact rectangular protection ROI instead:
    # the whole subject corridor stays on the temporally consistent video path,
    # while static guidance remains continuous everywhere outside that ROI.
    subject_evidence = np.logical_and(
        occupancy_gate < cfg.static_occupancy_barrier_threshold,
        motion_gate < cfg.static_occupancy_barrier_threshold,
    ).astype(np.uint8)
    subject_barrier = np.zeros_like(subject_evidence)
    if cfg.static_occupancy_barrier_radius > 0:
        radius = cfg.static_occupancy_barrier_radius
        subject_evidence = cv2.dilate(
            subject_evidence,
            np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
        )
    if face_box is not None and np.any(subject_evidence):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            subject_evidence, connectivity=8
        )
        x0, y0, x1, y1 = face_box
        seed_x0 = max(int(np.floor((x0 - cfg.static_face_expand_x) * grid_w)), 0)
        seed_x1 = min(int(np.ceil((x1 + cfg.static_face_expand_x) * grid_w)), grid_w)
        seed_y0 = max(int(np.floor((y0 - cfg.static_face_expand_top) * grid_h)), 0)
        seed_y1 = min(int(np.ceil((y1 + cfg.static_face_expand_bottom) * grid_h)), grid_h)
        seed_labels = labels[seed_y0:seed_y1, seed_x0:seed_x1]
        candidates = [int(label) for label in np.unique(seed_labels) if label > 0]
        if candidates:
            label = max(candidates, key=lambda value: int(stats[value, cv2.CC_STAT_AREA]))
            margin = max(int(cfg.static_subject_box_margin), 0)
            box_x = max(int(stats[label, cv2.CC_STAT_LEFT]) - margin, 0)
            box_y = max(int(stats[label, cv2.CC_STAT_TOP]) - margin, 0)
            box_w = int(stats[label, cv2.CC_STAT_WIDTH])
            box_h = int(stats[label, cv2.CC_STAT_HEIGHT])
            box_x1 = min(box_x + box_w + 2 * margin, grid_w)
            box_y1 = min(box_y + box_h + 2 * margin, grid_h)
            subject_barrier[box_y:box_y1, box_x:box_x1] = 1
    static_mask *= (1.0 - subject_barrier.astype(np.float32)) * foreground_gate
    if np.any(subject_barrier) and cfg.static_region_link_threshold_fraction > 0:
        threshold = (
            cfg.static_region_link_threshold_fraction * max(photo_range, cfg.eps)
        )
        pixel_count = grid_h * grid_w
        parents = np.arange(pixel_count, dtype=np.int32)

        def find(index: int) -> int:
            root = index
            while parents[root] != root:
                root = int(parents[root])
            while parents[index] != index:
                next_index = int(parents[index])
                parents[index] = root
                index = next_index
            return root

        def union(first: int, second: int) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                parents[second_root] = first_root

        for y in range(grid_h):
            row = y * grid_w
            for x in range(grid_w):
                index = row + x
                value = photo_grid[y, x]
                if not np.isfinite(value):
                    continue
                if x > 0 and abs(float(value - photo_grid[y, x - 1])) <= threshold:
                    union(index, index - 1)
                if y > 0 and abs(float(value - photo_grid[y - 1, x])) <= threshold:
                    union(index, index - grid_w)
        regions = np.fromiter(
            (find(index) for index in range(pixel_count)),
            dtype=np.int32,
            count=pixel_count,
        ).reshape(grid_h, grid_w)
        contaminated_regions = np.unique(regions[subject_barrier > 0])
        region_barrier = np.isin(regions, contaminated_regions)
        static_mask[region_barrier] = 0.0
    return (
        np.clip(static_mask, 0.0, 1.0).astype(np.float32),
        subject_barrier.astype(np.float32),
    )


def _residual_weight(distance: int, radius: int) -> float:
    if radius <= 0 or distance >= radius:
        return 0.0
    return float(0.5 * (1.0 + np.cos(np.pi * distance / radius)))


def _correction_edge_guard(
    base: np.ndarray,
    guidance_range: float,
    cfg: DepthSyncConfig,
    kernel_size: Optional[int] = None,
) -> np.ndarray:
    """Suppress coarse corrections around foreground/background discontinuities."""
    kernel_size = max(int(cfg.correction_edge_kernel if kernel_size is None else kernel_size) | 1, 1)
    if kernel_size <= 1 or guidance_range <= cfg.eps:
        return np.ones(base.shape, np.float32)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    finite = np.isfinite(base)
    fill = float(np.nanmedian(base)) if np.any(finite) else 0.0
    clean = np.nan_to_num(base, nan=fill, posinf=fill, neginf=fill).astype(np.float32)
    local_range = cv2.dilate(clean, kernel) - cv2.erode(clean, kernel)
    threshold = cfg.correction_edge_threshold_fraction * guidance_range
    softness = cfg.correction_edge_softness_fraction * guidance_range
    return np.clip((threshold - local_range) / max(softness, cfg.eps), 0.0, 1.0).astype(np.float32)


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


def _connected_depth_labels(depth: np.ndarray, threshold: float) -> np.ndarray:
    """Four-neighbour regions whose local disparity step stays below threshold."""
    height, width = depth.shape
    count = height * width
    parents = np.arange(count, dtype=np.int32)

    def find(index: int) -> int:
        root = index
        while parents[root] != root:
            root = int(parents[root])
        while parents[index] != index:
            next_index = int(parents[index])
            parents[index] = root
            index = next_index
        return root

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for y in range(height):
        row = y * width
        for x in range(width):
            index = row + x
            value = depth[y, x]
            if not np.isfinite(value):
                continue
            if x > 0 and abs(float(value - depth[y, x - 1])) <= threshold:
                union(index, index - 1)
            if y > 0 and abs(float(value - depth[y - 1, x])) <= threshold:
                union(index, index - width)
    roots = np.fromiter((find(i) for i in range(count)), np.int32, count=count)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.reshape(height, width).astype(np.int32)


def _largest_open_component(mask: np.ndarray, radius: int) -> np.ndarray:
    """Remove narrow region branches and retain the dominant connected plane."""
    if radius <= 0:
        return mask.astype(bool)
    opened = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if count <= 1:
        return np.zeros(mask.shape, bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _motion_global_shifts(
    motion: Optional[MotionSequence],
    frame_count: int,
    anchor_index: int,
    cfg: DepthSyncConfig,
) -> np.ndarray:
    """Current-frame to anchor normalized translations from robust block MV medians."""
    shifts = np.zeros((frame_count, 2), np.float32)
    if motion is None:
        return shifts

    def median_field(field: np.ndarray, confidence: Optional[np.ndarray]) -> np.ndarray:
        valid = np.all(np.isfinite(field), axis=-1)
        if confidence is not None:
            valid &= confidence >= cfg.region_motion_min_confidence
        if not np.any(valid):
            return np.zeros(2, np.float32)
        return np.median(field[valid], axis=0).astype(np.float32)

    for index in range(anchor_index + 1, frame_count):
        confidence = (
            None
            if motion.confidence_previous is None
            else motion.confidence_previous[index]
        )
        shifts[index] = shifts[index - 1] + median_field(
            motion.to_previous[index], confidence
        )
    for index in range(anchor_index - 1, -1, -1):
        confidence = (
            None if motion.confidence_next is None else motion.confidence_next[index]
        )
        shifts[index] = shifts[index + 1] + median_field(
            motion.to_next[index], confidence
        )
    return shifts


def _warp_region_labels(labels: np.ndarray, shift: np.ndarray) -> np.ndarray:
    height, width = labels.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x = xx + float(shift[0]) * max(width - 1, 1)
    map_y = yy + float(shift[1]) * max(height - 1, 1)
    return cv2.remap(
        labels,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _build_v4_regions(
    motion: Optional[MotionSequence],
    base_outputs: Sequence[np.ndarray],
    photo: np.ndarray,
    photo_range: float,
    anchor_index: int,
    face_box: Optional[tuple[float, float, float, float]],
    cfg: DepthSyncConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit photo-supervised scalar corrections on large, reliable video regions."""
    grid_h, grid_w = cfg.region_grid_shape
    frame_count = len(base_outputs)
    shifts = _motion_global_shifts(motion, frame_count, anchor_index, cfg)
    if motion is None or grid_h <= 0 or grid_w <= 0:
        return (
            np.zeros((0, 0), np.uint8),
            np.ones(1, np.float32),
            np.zeros((frame_count, 1), np.float32),
            shifts,
        )

    safe_mask, _subject_barrier = _static_guidance_mask(
        motion,
        base_outputs,
        photo,
        photo_range,
        (grid_h, grid_w),
        cfg,
        face_box,
    )
    anchor_grid = cv2.resize(
        base_outputs[anchor_index],
        (grid_w, grid_h),
        interpolation=cv2.INTER_AREA,
    )
    photo_grid = cv2.resize(photo, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    components = _connected_depth_labels(
        anchor_grid,
        cfg.region_link_threshold_fraction * max(photo_range, cfg.eps),
    )
    component_ids, component_areas = np.unique(components, return_counts=True)
    minimum_area = max(int(grid_h * grid_w * cfg.region_min_area_fraction), 1)
    order = component_ids[np.argsort(component_areas)[::-1]]
    region_labels = np.zeros((grid_h, grid_w), np.uint8)
    targets = [0.0]
    reliable_static = [False]
    face_region = np.zeros((grid_h, grid_w), bool)
    if face_box is not None:
        x0, y0, x1, y1 = face_box
        face_region[
            max(int(y0 * grid_h), 0) : min(int(np.ceil(y1 * grid_h)), grid_h),
            max(int(x0 * grid_w), 0) : min(int(np.ceil(x1 * grid_w)), grid_w),
        ] = True
    for component in order:
        pixels = components == component
        pixels = _largest_open_component(pixels, cfg.region_shape_open_radius)
        area = int(np.count_nonzero(pixels))
        if area < minimum_area or len(targets) > cfg.region_max_count:
            continue
        safe_support = float(np.mean(safe_mask[pixels] >= 0.5))
        # A constant correction attached to an anchor-frame subject region still
        # becomes a ghost when the subject moves away. V4 therefore stores only
        # regions supported as static over the full clip; every other pixel stays
        # on the global LUT path.
        if safe_support < cfg.region_static_support or np.any(face_region[pixels]):
            continue
        valid = pixels & np.isfinite(photo_grid) & np.isfinite(anchor_grid)
        if np.count_nonzero(valid) < minimum_area:
            continue
        label = len(targets)
        region_labels[pixels] = label
        targets.append(float(np.median(photo_grid[valid])))
        reliable_static.append(True)

    region_count = len(targets)
    scales = np.ones(region_count, np.float32)
    offsets = np.zeros((frame_count, region_count), np.float32)
    if region_count == 1:
        return region_labels, scales, offsets, shifts

    clip_limit = cfg.region_offset_clip_fraction * max(photo_range, cfg.eps)
    anchor_offsets = np.zeros(region_count, np.float32)
    for label in range(1, region_count):
        pixels = (region_labels == label) & np.isfinite(anchor_grid)
        if np.any(pixels):
            anchor_offsets[label] = float(
                np.clip(
                    targets[label] - np.median(anchor_grid[pixels]),
                    -clip_limit,
                    clip_limit,
                )
            )
            offsets[:, label] = anchor_offsets[label]
    for frame_index, base in enumerate(base_outputs):
        warped = _warp_region_labels(region_labels, shifts[frame_index])
        labels_full = cv2.resize(
            warped,
            (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        for label in range(1, region_count):
            if not reliable_static[label]:
                continue
            pixels = (labels_full == label) & np.isfinite(base)
            if np.any(pixels):
                offsets[frame_index, label] = float(
                    np.clip(targets[label] - np.median(base[pixels]), -clip_limit, clip_limit)
                )

    step_limit = cfg.region_offset_step_fraction * max(photo_range, cfg.eps)
    alpha = float(np.clip(cfg.region_offset_smoothing, 0.0, 1.0))
    for label in range(1, region_count):
        if not reliable_static[label]:
            continue
        for direction in (-1, 1):
            previous = float(offsets[anchor_index, label])
            for frame_index in range(
                anchor_index + direction,
                -1 if direction < 0 else frame_count,
                direction,
            ):
                candidate = (1.0 - alpha) * previous + alpha * float(
                    offsets[frame_index, label]
                )
                candidate = float(
                    np.clip(candidate, previous - step_limit, previous + step_limit)
                )
                offsets[frame_index, label] = candidate
                previous = candidate
    return region_labels, scales, offsets, shifts


def _apply_v4_regions(
    base: np.ndarray,
    labels: np.ndarray,
    scales: np.ndarray,
    offsets: np.ndarray,
    shift: np.ndarray,
    guidance_range: float,
    cfg: DepthSyncConfig,
) -> np.ndarray:
    if labels.size == 0 or len(scales) <= 1 or len(offsets) <= 1:
        return base
    warped = _warp_region_labels(labels, shift)
    labels_full = cv2.resize(
        warped,
        (base.shape[1], base.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    valid_labels = labels_full < min(len(scales), len(offsets))
    safe_labels = np.where(valid_labels, labels_full, 0)
    correction = (
        (scales[safe_labels] - 1.0) * base + offsets[safe_labels]
    ).astype(np.float32)
    edge_guard = _correction_edge_guard(
        base, guidance_range, cfg, cfg.region_edge_kernel
    )
    active = (safe_labels > 0).astype(np.float32)
    return (base + active * edge_guard * correction).astype(np.float32)


class DepthSync:
    """Estimate compact global-LUT and optional static-region parameters."""

    def __init__(self, config: Optional[DepthSyncConfig] = None):
        self.cfg = config or DepthSyncConfig()
        if self.cfg.mapping_mode not in {"affine", "lut"}:
            raise ValueError("mapping_mode must be 'affine' or 'lut'")
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
        if cfg.mapping_mode == "lut":
            candidate_x, candidate_y, lut_confidence = robust_monotonic_lut_samples(
                anchor_source, anchor_target, None, cfg
            )
            if lut_confidence > 0:
                anchor_lut_x, anchor_lut_y = _canonical_lut(
                    candidate_x, candidate_y, cfg.lut_nodes, cfg.eps
                )
                confidence = lut_confidence
        if confidence <= 0:
            reasons[anchor_index] = "insufficient_anchor_support"
        if cfg.mapping_mode == "lut":
            scales[anchor_index], offsets[anchor_index] = 1.0, 0.0
        else:
            scales[anchor_index], offsets[anchor_index] = a, b
        confidences[anchor_index] = confidence
        lut_x[anchor_index], lut_y[anchor_index] = anchor_lut_x, anchor_lut_y
        if cfg.mapping_mode == "lut":
            base_outputs[anchor_index] = _apply_lut(raw[anchor_index], anchor_lut_x, anchor_lut_y, cfg.eps)
        else:
            base_outputs[anchor_index] = (a * raw[anchor_index] + b).astype(np.float32)
        valid_photo = photo[np.isfinite(photo)]
        photo_range = float(np.quantile(valid_photo, 0.9) - np.quantile(valid_photo, 0.1)) if valid_photo.size else 1.0

        for direction in (-1, 1):
            previous_index = anchor_index
            for i in range(anchor_index + direction, -1 if direction < 0 else n, direction):
                # Global mapping propagates against the previous base map. The
                # spatial photo residual travels separately to avoid double use.
                previous = base_outputs[previous_index]
                assert previous is not None
                xs, ys = _sample_coordinates(raw[i], cfg, face_box)
                if cfg.mapping_mode == "lut":
                    pre_mapped = _apply_lut(raw[i], anchor_lut_x, anchor_lut_y, cfg.eps)
                else:
                    pre_mapped = raw[i]
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
                    if cfg.mapping_mode == "lut":
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
                    scale_delta = cfg.lut_max_scale_delta if cfg.mapping_mode == "lut" else cfg.max_scale_delta
                    offset_fraction = (
                        cfg.lut_max_offset_delta_fraction
                        if cfg.mapping_mode == "lut"
                        else cfg.max_offset_delta_fraction
                    )
                    a = float(np.clip(a, prev_a * (1.0 - scale_delta), prev_a * (1.0 + scale_delta)))
                    max_offset_delta = offset_fraction * max(photo_range, cfg.eps)
                    b = float(np.clip(b, prev_b - max_offset_delta, prev_b + max_offset_delta))
                    if cfg.mapping_mode == "lut":
                        current_x = anchor_lut_x
                        current_y = (a * anchor_lut_y + b).astype(np.float32)
                    else:
                        current_x, current_y = _affine_lut(source, a, b, cfg)
                if abs(i - anchor_index) <= cfg.anchor_lock_radius:
                    a, b = scales[anchor_index], offsets[anchor_index]
                    current_x, current_y = lut_x[anchor_index], lut_y[anchor_index]
                    reasons[i] = "anchor_lock"
                scales[i], offsets[i], confidences[i] = a, b, fit_conf
                lut_x[i], lut_y[i] = current_x, current_y
                if cfg.mapping_mode == "lut":
                    base_outputs[i] = _apply_lut(raw[i], current_x, current_y, cfg.eps)
                else:
                    base_outputs[i] = (a * raw[i] + b).astype(np.float32)
                previous_index = i

        if (
            cfg.mapping_mode == "lut"
            and face_box is not None
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

        if cfg.algorithm_version == "v4":
            concrete_base_outputs = [base for base in base_outputs if base is not None]
            assert len(concrete_base_outputs) == n
            region_labels, region_scales, region_offsets, region_shifts = (
                _build_v4_regions(
                    motion,
                    concrete_base_outputs,
                    photo,
                    photo_range,
                    anchor_index,
                    face_box,
                    cfg,
                )
            )
            outputs = [
                _apply_v4_regions(
                    base,
                    region_labels,
                    region_scales,
                    region_offsets[index],
                    region_shifts[index],
                    photo_range,
                    cfg,
                )
                for index, base in enumerate(concrete_base_outputs)
            ]
            empty_grid = np.zeros((n, 0, 0), np.float32)
            empty_shared = np.zeros((0, 0), np.float32)
            return SyncResult(
                np.stack(
                    [_from_working(x, cfg.depth_mode, cfg.eps) for x in outputs]
                ).astype(np.float32),
                scales.astype(np.float32),
                offsets.astype(np.float32),
                confidences.astype(np.float32),
                tuple(reasons),
                lut_x,
                lut_y,
                empty_grid,
                empty_shared,
                empty_shared.copy(),
                photo_range,
                region_labels,
                region_scales,
                region_offsets,
                region_shifts,
            )

        grid_h, grid_w = cfg.residual_grid_shape
        static_grid_h, static_grid_w = cfg.static_grid_shape
        target_grid_h, target_grid_w = cfg.static_target_shape
        residual_grids = np.zeros((n, max(grid_h, 0), max(grid_w, 0)), np.float32)
        concrete_base_outputs = [base for base in base_outputs if base is not None]
        assert len(concrete_base_outputs) == n
        static_mask, subject_barrier = _static_guidance_mask(
            motion,
            concrete_base_outputs,
            photo,
            photo_range,
            (static_grid_h, static_grid_w),
            cfg,
            face_box,
        )
        static_target_grid = (
            cv2.resize(photo, (target_grid_w, target_grid_h), interpolation=cv2.INTER_AREA).astype(np.float32)
            if static_grid_h > 0 and static_grid_w > 0 and target_grid_h > 0 and target_grid_w > 0
            else np.zeros((0, 0), np.float32)
        )
        if grid_h > 0 and grid_w > 0 and cfg.residual_radius > 0:
            anchor_base = base_outputs[anchor_index]
            assert anchor_base is not None
            anchor_grid = _make_residual_grid(photo - anchor_base, photo_range, cfg)
            residual_grids[anchor_index] = anchor_grid
            for direction in (-1, 1):
                transported = anchor_grid
                for i in range(anchor_index + direction, -1 if direction < 0 else n, direction):
                    distance = abs(i - anchor_index)
                    if distance >= cfg.residual_radius:
                        break
                    field = None
                    if motion is not None:
                        field = motion.to_previous[i] if direction > 0 else motion.to_next[i]
                    transported = _warp_residual_grid(transported, field, cfg.residual_motion_strength)
                    residual_grids[i] = transported * _residual_weight(distance, cfg.residual_radius)
            # Static cells use a clip-global photo target below. Suppress the
            # time-ramped residual there so the two corrections do not overlap.
            if static_mask.size:
                residual_static_mask = cv2.resize(
                    static_mask, (grid_w, grid_h), interpolation=cv2.INTER_AREA
                )
                residual_grids *= 1.0 - residual_static_mask[None]
            if subject_barrier.size:
                residual_subject_mask = cv2.resize(
                    subject_barrier,
                    (grid_w, grid_h),
                    interpolation=cv2.INTER_AREA,
                )
                residual_grids *= (
                    residual_subject_mask[None] <= cfg.eps
                ).astype(np.float32)

        outputs: list[np.ndarray] = []
        for base, residual_grid in zip(base_outputs, residual_grids):
            assert base is not None
            edge_guard = _correction_edge_guard(base, photo_range, cfg)
            if residual_grid.size:
                residual = cv2.resize(residual_grid, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
                base = base + edge_guard * residual
            if static_mask.size:
                mask = cv2.resize(static_mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
                target = cv2.resize(static_target_grid, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
                if cfg.static_correction_edge_kernel > 1:
                    mask *= _correction_edge_guard(
                        base, photo_range, cfg, cfg.static_correction_edge_kernel
                    )
                base = (1.0 - mask) * base + mask * target
            outputs.append(base.astype(np.float32))
        depths = np.stack([_from_working(x, cfg.depth_mode, cfg.eps) for x in outputs]).astype(np.float32)
        return SyncResult(
            depths,
            scales.astype(np.float32),
            offsets.astype(np.float32),
            confidences.astype(np.float32),
            tuple(reasons),
            lut_x,
            lut_y,
            residual_grids,
            static_mask,
            static_target_grid,
            photo_range,
            np.zeros((0, 0), np.uint8),
            np.ones(1, np.float32),
            np.zeros((n, 1), np.float32),
            np.zeros((n, 2), np.float32),
        )

    def apply_frame(
        self,
        depth: np.ndarray,
        params: FrameParameters,
        output_shape: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        working = _to_working(depth, self.cfg.depth_mode, self.cfg.eps)
        if self.cfg.mapping_mode == "lut" and params.lut_x is not None and params.lut_y is not None:
            synced = _apply_lut(working, params.lut_x, params.lut_y, self.cfg.eps)
        else:
            synced = params.scale * working + params.offset
        if (
            params.region_labels is not None
            and params.region_scales is not None
            and params.region_offsets is not None
            and params.region_shift is not None
        ):
            synced = _apply_v4_regions(
                synced,
                params.region_labels,
                params.region_scales,
                params.region_offsets,
                params.region_shift,
                params.guidance_range,
                self.cfg,
            )
        edge_guard = _correction_edge_guard(synced, params.guidance_range, self.cfg)
        if params.residual_grid is not None and params.residual_grid.size:
            residual = cv2.resize(
                params.residual_grid.astype(np.float32),
                (working.shape[1], working.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            synced = synced + edge_guard * residual
        if (
            params.static_mask is not None
            and params.static_mask.size
            and params.static_target_grid is not None
            and params.static_target_grid.size
        ):
            mask = cv2.resize(
                params.static_mask.astype(np.float32),
                (working.shape[1], working.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            if self.cfg.static_correction_edge_kernel > 1:
                mask *= _correction_edge_guard(
                    synced,
                    params.guidance_range,
                    self.cfg,
                    self.cfg.static_correction_edge_kernel,
                )
            target = cv2.resize(
                params.static_target_grid.astype(np.float32),
                (working.shape[1], working.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            synced = (1.0 - mask) * synced + mask * target
        if output_shape is not None and synced.shape[:2] != output_shape:
            synced = _resize(synced, output_shape, self.cfg.high_res_interpolation)
        return _from_working(synced, self.cfg.depth_mode, self.cfg.eps).astype(np.float32)

    def __call__(
        self,
        video_depths: Sequence[np.ndarray],
        photo_depth: np.ndarray,
        anchor_index: int,
        rgb_frames: Optional[Sequence[np.ndarray]] = None,
    ) -> SyncResult:
        # rgb_frames remains accepted for source compatibility; no dense flow is
        # computed by the edge path. Motion must be supplied explicitly.
        if rgb_frames is not None and len(rgb_frames) != len(video_depths):
            raise ValueError("rgb_frames and video_depths must have equal length")
        return self.offline_prepare(video_depths, photo_depth, anchor_index)
