"""Parameter-only photo-anchor depth synchronization for edge deployment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthSyncConfig:
    """Configuration for the lightweight parameter-estimation path."""

    depth_mode: str = "disparity"
    sample_count: int = 2048
    face_sample_fraction: float = 0.30
    edge_quantile: float = 0.70
    mad_threshold: float = 3.0
    parameter_smoothing: float = 0.65
    max_scale_delta: float = 0.03
    max_offset_delta_fraction: float = 0.02
    min_fit_pixels: int = 128
    high_res_interpolation: int = cv2.INTER_LINEAR
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


@dataclass
class SyncResult:
    depths: np.ndarray
    scales: np.ndarray
    offsets: np.ndarray
    confidences: np.ndarray
    fallback_reasons: tuple[str, ...]

    @property
    def parameters(self) -> tuple[FrameParameters, ...]:
        return tuple(
            FrameParameters(float(a), float(b), float(c), reason)
            for a, b, c, reason in zip(self.scales, self.offsets, self.confidences, self.fallback_reasons)
        )


def _valid(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & (x > 0)


def _resize(x: np.ndarray, shape: tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(x.astype(np.float32), (shape[1], shape[0]), interpolation=interpolation)


def _to_working(x: np.ndarray, mode: str, eps: float) -> np.ndarray:
    x = x.astype(np.float32)
    if mode == "disparity":
        return np.where(_valid(x), x, np.nan)
    if mode == "depth":
        return np.where(_valid(x), 1.0 / np.maximum(x, eps), np.nan)
    raise ValueError("depth_mode must be 'depth' or 'disparity'")


def _from_working(x: np.ndarray, mode: str, eps: float) -> np.ndarray:
    if mode == "depth":
        return np.where(np.isfinite(x) & (x > eps), 1.0 / np.maximum(x, eps), np.nan)
    return x


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


class DepthSync:
    """Estimate a tiny per-frame parameter table and apply it to video depth."""

    def __init__(self, config: Optional[DepthSyncConfig] = None):
        self.cfg = config or DepthSyncConfig()

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
        outputs: list[Optional[np.ndarray]] = [None] * n
        scales, offsets, confidences = np.ones(n), np.zeros(n), np.zeros(n)
        reasons = [""] * n

        xs, ys = _sample_coordinates(raw[anchor_index], cfg, face_box)
        anchor_source = _bilinear_sample(raw[anchor_index], xs, ys)
        anchor_target = _bilinear_sample(photo, xs, ys)
        a, b, confidence = robust_affine_samples(anchor_source, anchor_target, None, cfg)
        if confidence <= 0:
            reasons[anchor_index] = "insufficient_anchor_support"
        scales[anchor_index], offsets[anchor_index], confidences[anchor_index] = a, b, confidence
        outputs[anchor_index] = (a * raw[anchor_index] + b).astype(np.float32)
        valid_photo = photo[_valid(photo)]
        photo_range = float(np.quantile(valid_photo, 0.9) - np.quantile(valid_photo, 0.1)) if valid_photo.size else 1.0

        for direction in (-1, 1):
            previous_index = anchor_index
            for i in range(anchor_index + direction, -1 if direction < 0 else n, direction):
                previous = outputs[previous_index]
                assert previous is not None
                xs, ys = _sample_coordinates(raw[i], cfg, face_box)
                source = _bilinear_sample(raw[i], xs, ys)
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
                    reasons[i] = "insufficient_temporal_support"
                else:
                    alpha = float(np.clip(cfg.parameter_smoothing * fit_conf, 0.0, 1.0))
                    a = (1.0 - alpha) * prev_a + alpha * candidate_a
                    b = (1.0 - alpha) * prev_b + alpha * candidate_b
                    a = float(np.clip(a, prev_a * (1.0 - cfg.max_scale_delta), prev_a * (1.0 + cfg.max_scale_delta)))
                    max_offset_delta = cfg.max_offset_delta_fraction * max(photo_range, cfg.eps)
                    b = float(np.clip(b, prev_b - max_offset_delta, prev_b + max_offset_delta))
                scales[i], offsets[i], confidences[i] = a, b, fit_conf
                outputs[i] = (a * raw[i] + b).astype(np.float32)
                previous_index = i

        depths = np.stack([_from_working(x, cfg.depth_mode, cfg.eps) for x in outputs]).astype(np.float32)
        return SyncResult(depths, scales.astype(np.float32), offsets.astype(np.float32), confidences.astype(np.float32), tuple(reasons))

    def apply_frame(
        self,
        depth: np.ndarray,
        params: FrameParameters,
        output_shape: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        working = _to_working(depth, self.cfg.depth_mode, self.cfg.eps)
        synced = params.scale * working + params.offset
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
