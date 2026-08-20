"""Flow-guided local parameter alignment layered on top of the V4.1 LUT."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .flow import DenseFlowSequence, compose_to_anchor


@dataclass(frozen=True)
class LocalAlignmentConfig:
    grid_shape: tuple[int, int] = (8, 12)
    window_scale: float = 2.75
    min_fit_pixels: int = 96
    min_flow_confidence: float = 0.02
    min_improvement: float = 0.12
    scale_delta_bound: float = 0.08
    offset_bound_fraction: float = 0.12
    correction_bound_fraction: float = 0.16
    depth_sigma_fraction: float = 0.055
    temporal_transport_weight: float = 1.0
    max_scale_step: float = 0.012
    max_offset_step_fraction: float = 0.012
    max_confidence_step: float = 0.08
    distance_half_life: float = 90.0
    temporal_p95_ratio: float = 1.15
    temporal_p95_budget: float = 0.00025
    temporal_max_excess: float = 0.002
    strength_candidates: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
    eps: float = 1e-6


@dataclass(frozen=True)
class LocalFieldFrame:
    delta_scale: np.ndarray
    offset_norm: np.ndarray
    confidence: np.ndarray
    guide_depth: np.ndarray
    photo_range: float


@dataclass(frozen=True)
class LocalFieldSequence:
    delta_scale: np.ndarray
    offset_norm: np.ndarray
    confidence: np.ndarray
    guide_depth: np.ndarray
    photo_range: float

    def __post_init__(self) -> None:
        shape = self.delta_scale.shape
        if len(shape) != 3 or any(
            values.shape != shape
            for values in (self.offset_norm, self.confidence, self.guide_depth)
        ):
            raise ValueError("local field channels must have equal [T,Gh,Gw] shapes")
        if not np.isfinite(self.photo_range) or self.photo_range <= 0.0:
            raise ValueError("photo_range must be finite and positive")
        for values in (self.delta_scale, self.offset_norm, self.confidence, self.guide_depth):
            if not np.all(np.isfinite(values)):
                raise ValueError("local field channels must be finite")
        if np.any((self.confidence < 0.0) | (self.confidence > 1.0)):
            raise ValueError("local field confidence must be within [0,1]")

    def frame(self, index: int) -> LocalFieldFrame:
        return LocalFieldFrame(
            self.delta_scale[index],
            self.offset_norm[index],
            self.confidence[index],
            self.guide_depth[index],
            self.photo_range,
        )


def _robust_range(values: np.ndarray, eps: float) -> float:
    finite = np.asarray(values, np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    return max(float(np.quantile(finite, 0.9) - np.quantile(finite, 0.1)), eps)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = 0.5 * float(np.sum(ordered_weights))
    return float(ordered_values[min(int(np.searchsorted(np.cumsum(ordered_weights), cutoff)), len(values) - 1)])


def _fit_local_model(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    photo_range: float,
    config: LocalAlignmentConfig,
) -> tuple[float, float, float]:
    valid = np.isfinite(source) & np.isfinite(target) & np.isfinite(weights) & (weights > 0.0)
    if int(np.count_nonzero(valid)) < config.min_fit_pixels:
        return 0.0, 0.0, 0.0
    x = source[valid].astype(np.float64)
    y = target[valid].astype(np.float64)
    w = weights[valid].astype(np.float64)
    w /= max(float(np.sum(w)), config.eps)
    identity_error = float(np.sum(w * np.abs(y - x)))
    if identity_error < 0.006 * photo_range:
        return 0.0, 0.0, 0.0

    scale = 1.0
    offset = _weighted_median(y - x, w)
    for _ in range(3):
        residual = y - (scale * x + offset)
        center = _weighted_median(residual, w)
        mad = _weighted_median(np.abs(residual - center), w)
        robust = max(1.4826 * mad, 0.01 * photo_range, config.eps)
        inlier_weight = w * np.square(np.clip(1.0 - np.square((residual - center) / (4.685 * robust)), 0.0, 1.0))
        total = float(np.sum(inlier_weight))
        if total <= config.eps:
            return 0.0, 0.0, 0.0
        inlier_weight /= total
        mean_x = float(np.sum(inlier_weight * x))
        mean_y = float(np.sum(inlier_weight * y))
        centered_x = x - mean_x
        variance = float(np.sum(inlier_weight * centered_x * centered_x))
        if variance >= (0.025 * photo_range) ** 2:
            covariance = float(np.sum(inlier_weight * centered_x * (y - mean_y)))
            scale_ridge = (0.025 * photo_range) ** 2
            raw_scale = (covariance + scale_ridge) / (variance + scale_ridge)
            scale = float(np.clip(raw_scale, 1.0 - config.scale_delta_bound, 1.0 + config.scale_delta_bound))
        else:
            scale = 1.0
        offset = float(
            np.clip(
                _weighted_median(y - scale * x, inlier_weight),
                -config.offset_bound_fraction * photo_range,
                config.offset_bound_fraction * photo_range,
            )
        )

    fitted_error = float(np.sum(w * np.abs(y - (scale * x + offset))))
    improvement = float(np.clip(1.0 - fitted_error / max(identity_error, config.eps), 0.0, 1.0))
    confidence_floor = min(0.02, 0.5 * config.min_improvement)
    if improvement < confidence_floor:
        return 0.0, 0.0, 0.0
    residual_consistency = float(np.exp(-fitted_error / (0.08 * photo_range + config.eps)))
    confidence = np.clip(
        (improvement - confidence_floor) / max(0.45 - confidence_floor, config.eps),
        0.0,
        1.0,
    ) * residual_consistency
    return scale - 1.0, offset / photo_range, float(confidence)


def _fit_frame(
    base: np.ndarray,
    target: np.ndarray,
    track_confidence: np.ndarray,
    photo_range: float,
    config: LocalAlignmentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = base.shape
    gh, gw = config.grid_shape
    delta_scale = np.zeros((gh, gw), np.float32)
    offset_norm = np.zeros((gh, gw), np.float32)
    confidence = np.zeros((gh, gw), np.float32)
    guide_depth = cv2.resize(base, (gw, gh), cv2.INTER_AREA).astype(np.float32)
    window_h = max(7, int(round(config.window_scale * height / gh)))
    window_w = max(7, int(round(config.window_scale * width / gw)))
    gradient = cv2.magnitude(
        cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3),
    )
    edge_weight = np.exp(-gradient / (0.10 * photo_range + config.eps)).astype(np.float32)

    for gy in range(gh):
        center_y = (gy + 0.5) * height / gh
        y0 = max(0, int(round(center_y - window_h / 2)))
        y1 = min(height, y0 + window_h)
        y0 = max(0, y1 - window_h)
        for gx in range(gw):
            center_x = (gx + 0.5) * width / gw
            x0 = max(0, int(round(center_x - window_w / 2)))
            x1 = min(width, x0 + window_w)
            x0 = max(0, x1 - window_w)
            source = base[y0:y1, x0:x1]
            reference = target[y0:y1, x0:x1]
            flow_weight = track_confidence[y0:y1, x0:x1]
            valid = np.isfinite(source) & np.isfinite(reference) & (flow_weight >= config.min_flow_confidence)
            if int(np.count_nonzero(valid)) < config.min_fit_pixels:
                continue
            local_values = source[valid]
            node_depth = float(np.median(local_values))
            guide_depth[gy, gx] = node_depth
            depth_weight = np.exp(
                -np.abs(source - node_depth) / (config.depth_sigma_fraction * photo_range + config.eps)
            )
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            spatial = np.exp(
                -0.5 * (
                    np.square((xx - center_x) / max(0.45 * window_w, 1.0))
                    + np.square((yy - center_y) / max(0.45 * window_h, 1.0))
                )
            )
            weights = flow_weight * depth_weight * spatial * edge_weight[y0:y1, x0:x1]
            weights = np.where(valid, weights, 0.0).astype(np.float32)
            ds, off, fit_confidence = _fit_local_model(source, reference, weights, photo_range, config)
            support = min(1.0, float(np.count_nonzero(weights > 0.05)) / max(0.35 * weights.size, 1.0))
            flow_score = float(np.median(flow_weight[valid]))
            node_confidence = fit_confidence * support * flow_score
            if node_confidence > 0.02:
                delta_scale[gy, gx] = ds
                offset_norm[gy, gx] = off
                confidence[gy, gx] = node_confidence
    return delta_scale, offset_norm, confidence, guide_depth


def _resize_current_to_nearer(
    flow_values: np.ndarray,
    confidence: np.ndarray,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    source_h, source_w = flow_values.shape[:2]
    gh, gw = grid_shape
    field = cv2.resize(flow_values.astype(np.float32), (gw, gh), cv2.INTER_LINEAR)
    field[..., 0] *= gw / max(source_w, 1)
    field[..., 1] *= gh / max(source_h, 1)
    reliability = cv2.resize(confidence.astype(np.float32), (gw, gh), cv2.INTER_LINEAR)
    return field, np.clip(reliability, 0.0, 1.0)


def _transport(values: np.ndarray, field: np.ndarray) -> np.ndarray:
    gh, gw = values.shape[:2]
    yy, xx = np.mgrid[:gh, :gw].astype(np.float32)
    return cv2.remap(
        values.astype(np.float32),
        xx + field[..., 0],
        yy + field[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def fit_flow_guided_fields(
    base_depths: np.ndarray,
    photo_anchor: np.ndarray,
    anchor_index: int,
    flow: DenseFlowSequence,
    config: LocalAlignmentConfig | None = None,
) -> LocalFieldSequence:
    """Fit conservative per-frame local parameters from flow correspondences."""
    cfg = config or LocalAlignmentConfig()
    base = np.asarray(base_depths, np.float32)
    photo = np.asarray(photo_anchor, np.float32)
    if base.ndim != 3 or photo.shape != base.shape[1:]:
        raise ValueError("base_depths and photo_anchor shapes do not match")
    if flow.frame_count != len(base):
        raise ValueError("flow and base_depths must have equal frame counts")
    if not 0 <= anchor_index < len(base):
        raise ValueError("anchor_index is out of range")
    photo_range = _robust_range(photo, cfg.eps)
    coordinates, track_confidence = compose_to_anchor(
        flow, anchor_index, cfg.min_flow_confidence * 0.25, cfg.distance_half_life
    )
    frame_count, height, width = base.shape
    gh, gw = cfg.grid_shape
    delta_scale = np.zeros((frame_count, gh, gw), np.float32)
    offset_norm = np.zeros_like(delta_scale)
    confidence = np.zeros_like(delta_scale)
    guide_depth = np.zeros_like(delta_scale)

    for index in range(frame_count):
        normalized_map = cv2.resize(coordinates[index], (width, height), cv2.INTER_LINEAR)
        reliability = cv2.resize(track_confidence[index], (width, height), cv2.INTER_LINEAR)
        target = cv2.remap(
            photo,
            normalized_map[..., 0] * max(width - 1, 1),
            normalized_map[..., 1] * max(height - 1, 1),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
        ds, off, conf, guide = _fit_frame(base[index], target, reliability, photo_range, cfg)
        delta_scale[index], offset_norm[index], confidence[index], guide_depth[index] = ds, off, conf, guide

    # A symmetric offline solve avoids direction-dependent breathing.  Neighbor
    # parameters are first transported into the current frame.  Nodes without
    # direct current-frame evidence remain exact identity, so revealed areas
    # are never filled solely from an anchor trajectory.
    direct_support = confidence > 0.0
    for _ in range(5):
        next_scale = delta_scale.copy()
        next_offset = offset_norm.copy()
        next_confidence = confidence.copy()
        for index in range(frame_count):
            if index == anchor_index:
                continue
            self_weight = confidence[index].copy()
            scale_sum = self_weight * delta_scale[index]
            offset_sum = self_weight * offset_norm[index]
            confidence_sum = self_weight * confidence[index]
            total = self_weight.copy()
            neighbors: list[tuple[int, np.ndarray, np.ndarray, bool]] = []
            if index > 0:
                pair = index - 1
                neighbors.append(
                    (pair, flow.to_previous[pair], flow.confidence_previous[pair], bool(flow.scene_cuts[pair]))
                )
            if index + 1 < frame_count:
                pair = index
                neighbors.append(
                    (pair, flow.to_next[pair], flow.confidence_next[pair], bool(flow.scene_cuts[pair]))
                )
            for pair, pair_flow, pair_confidence, is_cut in neighbors:
                neighbor = pair if pair < index else pair + 1
                field, reliability = _resize_current_to_nearer(
                    pair_flow, pair_confidence, cfg.grid_shape
                )
                if is_cut:
                    reliability.fill(0.0)
                transported_confidence = _transport(confidence[neighbor], field)
                weight = cfg.temporal_transport_weight * reliability * transported_confidence
                weight *= direct_support[index]
                scale_sum += weight * _transport(delta_scale[neighbor], field)
                offset_sum += weight * _transport(offset_norm[neighbor], field)
                confidence_sum += weight * transported_confidence
                total += weight
            supported = total > cfg.eps
            next_scale[index] = np.where(supported, scale_sum / np.maximum(total, cfg.eps), 0.0)
            next_offset[index] = np.where(supported, offset_sum / np.maximum(total, cfg.eps), 0.0)
            next_confidence[index] = np.where(
                direct_support[index] & supported,
                confidence_sum / np.maximum(total, cfg.eps),
                0.0,
            )
        delta_scale, offset_norm, confidence = next_scale, next_offset, np.clip(next_confidence, 0.0, 1.0)

    # Bound the remaining trajectory-wise parameter velocity.  This limiter is
    # applied radially from the exact anchor and therefore cannot accumulate a
    # one-frame activation into a visible depth jump.
    for direction in (-1, 1):
        stop = -1 if direction < 0 else frame_count
        for index in range(anchor_index + direction, stop, direction):
            nearer = index - direction
            pair = index if direction < 0 else index - 1
            if direction < 0:
                pair_flow, pair_confidence = flow.to_next[pair], flow.confidence_next[pair]
            else:
                pair_flow, pair_confidence = flow.to_previous[pair], flow.confidence_previous[pair]
            field, reliability = _resize_current_to_nearer(
                pair_flow, pair_confidence, cfg.grid_shape
            )
            if flow.scene_cuts[pair]:
                reliability.fill(0.0)
            transported_confidence = _transport(confidence[nearer], field)
            trusted = direct_support[index] & (transported_confidence > 0.0) & (reliability > 0.10)
            transported_scale = _transport(delta_scale[nearer], field)
            transported_offset = _transport(offset_norm[nearer], field)
            delta_scale[index] = np.where(
                trusted,
                np.clip(
                    delta_scale[index],
                    transported_scale - cfg.max_scale_step,
                    transported_scale + cfg.max_scale_step,
                ),
                delta_scale[index],
            )
            offset_norm[index] = np.where(
                trusted,
                np.clip(
                    offset_norm[index],
                    transported_offset - cfg.max_offset_step_fraction,
                    transported_offset + cfg.max_offset_step_fraction,
                ),
                offset_norm[index],
            )
            confidence[index] = np.where(
                trusted,
                np.clip(
                    confidence[index],
                    np.maximum(transported_confidence - cfg.max_confidence_step, 0.0),
                    np.minimum(transported_confidence + cfg.max_confidence_step, 1.0),
                ),
                confidence[index],
            )

    delta_scale = np.clip(delta_scale, -cfg.scale_delta_bound, cfg.scale_delta_bound)
    offset_norm = np.clip(offset_norm, -cfg.offset_bound_fraction, cfg.offset_bound_fraction)
    return LocalFieldSequence(delta_scale, offset_norm, confidence, guide_depth, photo_range)


def apply_local_field(
    base: np.ndarray,
    field: LocalFieldFrame,
    config: LocalAlignmentConfig | None = None,
) -> np.ndarray:
    """Apply one edge-aware local field, exactly preserving unsupported pixels."""
    cfg = config or LocalAlignmentConfig(grid_shape=field.confidence.shape)
    source = np.asarray(base, np.float32)
    if source.ndim != 2:
        raise ValueError("base must have shape [H,W]")
    if float(np.max(field.confidence, initial=0.0)) <= 0.0:
        return source.copy()
    height, width = source.shape
    gh, gw = field.confidence.shape
    grid_x = (np.arange(width, dtype=np.float32) + 0.5) * gw / width - 0.5
    grid_y = (np.arange(height, dtype=np.float32) + 0.5) * gh / height - 0.5
    x0 = np.floor(grid_x).astype(np.int32)
    y0 = np.floor(grid_y).astype(np.int32)
    tx, ty = grid_x - x0, grid_y - y0
    numerator_scale = np.zeros_like(source)
    numerator_offset = np.zeros_like(source)
    supported = np.zeros_like(source)
    possible = np.zeros_like(source)
    sigma = cfg.depth_sigma_fraction * _robust_range(source, cfg.eps)
    safe_source = np.nan_to_num(source, nan=0.0)
    for yi, wy in ((np.clip(y0, 0, gh - 1), 1.0 - ty), (np.clip(y0 + 1, 0, gh - 1), ty)):
        for xi, wx in ((np.clip(x0, 0, gw - 1), 1.0 - tx), (np.clip(x0 + 1, 0, gw - 1), tx)):
            spatial = wy[:, None] * wx[None, :]
            node_depth = field.guide_depth[yi[:, None], xi[None, :]]
            guide = np.exp(np.clip(-np.abs(safe_source - node_depth) / max(sigma, cfg.eps), -50.0, 0.0))
            available = spatial * guide
            node_confidence = field.confidence[yi[:, None], xi[None, :]]
            weight = available * node_confidence
            possible += available
            supported += weight
            numerator_scale += weight * field.delta_scale[yi[:, None], xi[None, :]]
            numerator_offset += weight * field.offset_norm[yi[:, None], xi[None, :]]
    valid = supported > cfg.eps
    ds = np.where(valid, numerator_scale / np.maximum(supported, cfg.eps), 0.0)
    off = np.where(valid, numerator_offset / np.maximum(supported, cfg.eps), 0.0)
    weight = np.where(possible > cfg.eps, supported / np.maximum(possible, cfg.eps), 0.0)
    correction = ds * source + off * field.photo_range
    correction = np.clip(
        correction,
        -cfg.correction_bound_fraction * field.photo_range,
        cfg.correction_bound_fraction * field.photo_range,
    )
    return (source + weight * correction).astype(np.float32)


def apply_local_sequence(
    base_depths: np.ndarray,
    fields: LocalFieldSequence,
    config: LocalAlignmentConfig | None = None,
) -> np.ndarray:
    if len(base_depths) != len(fields.delta_scale):
        raise ValueError("base_depths and fields must have equal frame counts")
    cfg = config or LocalAlignmentConfig(grid_shape=fields.confidence.shape[1:])
    return np.stack(
        [apply_local_field(depth, fields.frame(index), cfg) for index, depth in enumerate(base_depths)]
    ).astype(np.float32)


def dense_temporal_errors(
    sequence: np.ndarray,
    flow: DenseFlowSequence,
    scale: float,
) -> np.ndarray:
    """Motion-compensated temporal errors using the same dense flow as fitting."""
    values = np.asarray(sequence, np.float32)
    if values.ndim != 3 or len(values) != flow.frame_count:
        raise ValueError("sequence and flow must have equal frame counts")
    height, width = values.shape[1:]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    errors = np.full(len(values) - 1, np.nan, np.float32)
    flow_h, flow_w = flow.to_previous.shape[1:3]
    for index in range(1, len(values)):
        field = cv2.resize(flow.to_previous[index - 1], (width, height), cv2.INTER_LINEAR)
        field[..., 0] *= width / max(flow_w, 1)
        field[..., 1] *= height / max(flow_h, 1)
        previous = cv2.remap(
            values[index - 1],
            xx + field[..., 0],
            yy + field[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
        confidence = cv2.resize(
            flow.confidence_previous[index - 1], (width, height), cv2.INTER_LINEAR
        )
        valid = np.isfinite(previous) & np.isfinite(values[index]) & (confidence > 0.20)
        if np.any(valid):
            errors[index - 1] = float(
                np.median(np.abs(values[index][valid] - previous[valid])) / max(scale, 1e-6)
            )
    return errors


def select_temporal_safe_fields(
    base_depths: np.ndarray,
    fields: LocalFieldSequence,
    flow: DenseFlowSequence,
    config: LocalAlignmentConfig | None = None,
    photo_anchor: np.ndarray | None = None,
    anchor_index: int | None = None,
    priority_box: tuple[float, float, float, float] | None = None,
) -> tuple[LocalFieldSequence, float, dict[str, float]]:
    """Choose the strongest global field gain that stays inside a flow budget."""
    cfg = config or LocalAlignmentConfig(grid_shape=fields.confidence.shape[1:])
    base_errors = dense_temporal_errors(base_depths, flow, fields.photo_range)
    interior = slice(4, max(len(base_errors) - 5, 5))
    base_p95 = float(np.nanquantile(base_errors[interior], 0.95))
    selected = fields
    selected_strength = 0.0
    selected_p95 = base_p95
    selected_max_excess = 0.0
    for strength in cfg.strength_candidates:
        candidate = LocalFieldSequence(
            fields.delta_scale,
            fields.offset_norm,
            np.clip(fields.confidence * strength, 0.0, 1.0).astype(np.float32),
            fields.guide_depth,
            fields.photo_range,
        )
        output = apply_local_sequence(base_depths, candidate, cfg)
        errors = dense_temporal_errors(output, flow, fields.photo_range)
        p95 = float(np.nanquantile(errors[interior], 0.95))
        max_excess = float(np.nanmax(errors[interior] - base_errors[interior]))
        p95_limit = base_p95 * cfg.temporal_p95_ratio + cfg.temporal_p95_budget
        if p95 <= p95_limit and max_excess <= cfg.temporal_max_excess:
            selected = candidate
            selected_strength = float(strength)
            selected_p95 = p95
            selected_max_excess = max_excess
            break
    diagnostics = {
        "dense_base_temporal_p95": base_p95,
        "dense_local_temporal_p95": selected_p95,
        "dense_local_max_excess": selected_max_excess,
    }
    if photo_anchor is not None and anchor_index is not None and priority_box is not None:
        coordinates, _ = compose_to_anchor(flow, anchor_index, 0.0, cfg.distance_half_life)
        gh, gw = fields.confidence.shape[1:]
        priority = np.zeros_like(fields.confidence, np.float32)
        x0, y0, x1, y1 = priority_box
        protected_x0 = max(0.0, x0 - 1.0 / max(gw, 1))
        protected_y0 = max(0.0, y0 - 1.0 / max(gh, 1))
        protected_x1 = min(1.0, x1 + 1.0 / max(gw, 1))
        protected_y1 = min(1.0, y1 + 1.0 / max(gh, 1))
        for index in range(len(priority)):
            grid = cv2.resize(coordinates[index], (gw, gh), cv2.INTER_LINEAR)
            priority[index] = (
                (grid[..., 0] >= protected_x0)
                & (grid[..., 0] <= protected_x1)
                & (grid[..., 1] >= protected_y0)
                & (grid[..., 1] <= protected_y1)
            ).astype(np.float32)

        def priority_error(sequence: np.ndarray) -> float:
            height, width = sequence.shape[1:]
            flow_h, flow_w = flow.to_next.shape[1:3]
            yy, xx = np.mgrid[:height, :width].astype(np.float32)
            errors: list[float] = []
            for index, pair_values, pair_confidence in (
                (anchor_index - 1, flow.to_next[anchor_index - 1], flow.confidence_next[anchor_index - 1]),
                (anchor_index + 1, flow.to_previous[anchor_index], flow.confidence_previous[anchor_index]),
            ):
                if not 0 <= index < len(sequence):
                    continue
                pair = cv2.resize(pair_values, (width, height), cv2.INTER_LINEAR)
                pair[..., 0] *= width / max(flow_w, 1)
                pair[..., 1] *= height / max(flow_h, 1)
                map_x = xx + pair[..., 0]
                map_y = yy + pair[..., 1]
                target = cv2.remap(
                    photo_anchor.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
                )
                reliability = cv2.resize(pair_confidence, (width, height), cv2.INTER_LINEAR)
                valid = (
                    np.isfinite(target)
                    & np.isfinite(sequence[index])
                    & (reliability > 0.20)
                    & (map_x >= x0 * width)
                    & (map_x < x1 * width)
                    & (map_y >= y0 * height)
                    & (map_y < y1 * height)
                )
                if np.any(valid):
                    errors.append(float(np.median(np.abs(sequence[index][valid] - target[valid])) / fields.photo_range))
            return float(np.mean(errors)) if errors else float("inf")

        base_priority_error = priority_error(np.asarray(base_depths, np.float32))
        selected_priority_strength = 0.0
        selected_priority_error = base_priority_error
        for priority_strength in (
            strength for strength in cfg.strength_candidates if strength <= selected_strength
        ):
            gain = selected_strength * (1.0 - priority) + priority_strength * priority
            candidate = LocalFieldSequence(
                fields.delta_scale,
                fields.offset_norm,
                np.clip(fields.confidence * gain, 0.0, 1.0).astype(np.float32),
                fields.guide_depth,
                fields.photo_range,
            )
            error = priority_error(apply_local_sequence(base_depths, candidate, cfg))
            if error <= base_priority_error * 1.01 + 0.0002:
                selected = candidate
                selected_priority_strength = float(priority_strength)
                selected_priority_error = error
                break
        diagnostics.update(
            {
                "priority_base_switch_error": base_priority_error,
                "priority_local_switch_error": selected_priority_error,
                "priority_strength": selected_priority_strength,
            }
        )
    return selected, selected_strength, diagnostics


def save_local_fields(path: str, fields: LocalFieldSequence) -> None:
    np.savez_compressed(
        path,
        delta_scale=fields.delta_scale.astype(np.float16),
        offset_norm=fields.offset_norm.astype(np.float16),
        confidence=fields.confidence.astype(np.float16),
        guide_depth=fields.guide_depth.astype(np.float16),
        photo_range=np.float32(fields.photo_range),
    )
