"""Lightweight photo-anchor guided temporal depth synchronization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthSyncConfig:
    """Tunable parameters. Depth is converted to disparity before processing by default."""

    depth_mode: str = "depth"  # "depth" (metric-like) or "disparity" (larger=nearer)
    temporal_strength: float = 0.72
    detail_strength: float = 0.35
    detail_sigma: float = 5.0
    flow_levels: int = 3
    flow_winsize: int = 21
    min_fit_pixels: int = 256
    eps: float = 1e-6


@dataclass
class SyncResult:
    depths: np.ndarray
    scales: np.ndarray
    offsets: np.ndarray
    confidences: np.ndarray


def _valid(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & (x > 0)


def _resize(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(x.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _to_working(x: np.ndarray, mode: str, eps: float) -> np.ndarray:
    x = x.astype(np.float32)
    if mode == "depth":
        return np.where(_valid(x), 1.0 / np.maximum(x, eps), np.nan)
    if mode == "disparity":
        return np.where(_valid(x), x, np.nan)
    raise ValueError("depth_mode must be 'depth' or 'disparity'")


def _from_working(x: np.ndarray, mode: str, eps: float) -> np.ndarray:
    if mode == "depth":
        return np.where(np.isfinite(x) & (x > eps), 1.0 / np.maximum(x, eps), np.nan)
    return x


def robust_affine(source: np.ndarray, target: np.ndarray, mask: np.ndarray, min_pixels: int) -> tuple[float, float, float]:
    """Fit target ~= scale * source + offset using trimmed IRLS."""
    good = mask & np.isfinite(source) & np.isfinite(target)
    x, y = source[good].astype(np.float64), target[good].astype(np.float64)
    if x.size < min_pixels or np.nanstd(x) < 1e-8:
        return 1.0, 0.0, 0.0
    # Quantile initialization is robust to differing resolution and edge outliers.
    qx = np.quantile(x, [0.1, 0.5, 0.9])
    qy = np.quantile(y, [0.1, 0.5, 0.9])
    scale = max((qy[2] - qy[0]) / max(qx[2] - qx[0], 1e-8), 1e-6)
    offset = qy[1] - scale * qx[1]
    a = np.column_stack((x, np.ones_like(x)))
    for _ in range(5):
        residual = y - (scale * x + offset)
        sigma = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-8
        w = np.minimum(1.0, 1.5 * sigma / np.maximum(np.abs(residual), 1e-8))
        lhs = a.T @ (w[:, None] * a)
        rhs = a.T @ (w * y)
        try:
            scale, offset = np.linalg.solve(lhs + np.eye(2) * 1e-8, rhs)
        except np.linalg.LinAlgError:
            break
    scale = float(np.clip(scale, 1e-4, 1e4))
    residual = y - (scale * x + offset)
    confidence = float(np.clip(1.0 - np.median(np.abs(residual)) / (np.std(y) + 1e-8), 0.0, 1.0))
    return scale, float(offset), confidence


def _gray(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    f = _resize(frame, shape) if frame.shape[:2] != shape else frame
    if f.ndim == 3:
        f = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    f = f.astype(np.float32)
    if f.max() > 1.5:
        f /= 255.0
    return f


def _warp_previous(previous: np.ndarray, previous_rgb: np.ndarray, current_rgb: np.ndarray, cfg: DepthSyncConfig) -> tuple[np.ndarray, np.ndarray]:
    shape = previous.shape
    gp, gc = _gray(previous_rgb, shape), _gray(current_rgb, shape)
    # Backward flow (current -> previous) directly supplies remap coordinates.
    back = cv2.calcOpticalFlowFarneback(gc, gp, None, 0.5, cfg.flow_levels, cfg.flow_winsize, 3, 5, 1.2, 0)
    forward = cv2.calcOpticalFlowFarneback(gp, gc, None, 0.5, cfg.flow_levels, cfg.flow_winsize, 3, 5, 1.2, 0)
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    mapx, mapy = xx + back[..., 0], yy + back[..., 1]
    warped = cv2.remap(previous, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    fw_at_current = cv2.remap(forward, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    fb_error = np.linalg.norm(back + fw_at_current, axis=2)
    photo_error = np.abs(gc - cv2.remap(gp, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101))
    confidence = np.exp(-0.7 * fb_error - 6.0 * photo_error).astype(np.float32)
    inside = (mapx >= 0) & (mapx < shape[1] - 1) & (mapy >= 0) & (mapy < shape[0] - 1)
    return warped, confidence * inside


class DepthSync:
    """Synchronize a low-resolution video depth sequence to one high-quality photo depth."""

    def __init__(self, config: Optional[DepthSyncConfig] = None):
        self.cfg = config or DepthSyncConfig()

    def __call__(self, video_depths: Sequence[np.ndarray], photo_depth: np.ndarray, anchor_index: int, rgb_frames: Optional[Sequence[np.ndarray]] = None) -> SyncResult:
        if len(video_depths) == 0:
            raise ValueError("video_depths is empty")
        if not 0 <= anchor_index < len(video_depths):
            raise ValueError("anchor_index is out of range")
        if rgb_frames is not None and len(rgb_frames) != len(video_depths):
            raise ValueError("rgb_frames and video_depths must have equal length")
        cfg, shape, n = self.cfg, photo_depth.shape[:2], len(video_depths)
        photo = _to_working(photo_depth, cfg.depth_mode, cfg.eps)
        raw = [_resize(_to_working(d, cfg.depth_mode, cfg.eps), shape) for d in video_depths]
        outputs: list[Optional[np.ndarray]] = [None] * n
        scales, offsets, confidences = np.ones(n), np.zeros(n), np.zeros(n)

        anchor_mask = _valid(raw[anchor_index]) & _valid(photo)
        a, b, c = robust_affine(raw[anchor_index], photo, anchor_mask, cfg.min_fit_pixels)
        anchor_base = a * raw[anchor_index] + b
        # Keep the high-frequency photo residual while avoiding copying all photo noise.
        smooth_photo = cv2.GaussianBlur(np.nan_to_num(photo, nan=float(np.nanmedian(photo))), (0, 0), cfg.detail_sigma)
        smooth_base = cv2.GaussianBlur(np.nan_to_num(anchor_base, nan=float(np.nanmedian(anchor_base))), (0, 0), cfg.detail_sigma)
        detail = (photo - smooth_photo) - (anchor_base - smooth_base)
        outputs[anchor_index] = np.where(anchor_mask, anchor_base + cfg.detail_strength * detail, photo).astype(np.float32)
        scales[anchor_index], offsets[anchor_index], confidences[anchor_index] = a, b, c

        for direction in (-1, 1):
            indices = range(anchor_index + direction, -1 if direction < 0 else n, direction)
            previous_index = anchor_index
            for i in indices:
                previous = outputs[previous_index]
                assert previous is not None
                if rgb_frames is not None:
                    warped, flow_conf = _warp_previous(previous, rgb_frames[previous_index], rgb_frames[i], cfg)
                else:
                    warped, flow_conf = previous, np.full(shape, 0.45, np.float32)
                gradient = np.hypot(*np.gradient(warped))
                stable = flow_conf > 0.15
                if np.any(stable):
                    stable &= gradient < np.quantile(gradient[stable], 0.8)
                a, b, fit_conf = robust_affine(raw[i], warped, stable & _valid(raw[i]) & _valid(warped), cfg.min_fit_pixels)
                calibrated = a * raw[i] + b
                disagreement = np.abs(calibrated - warped) / (np.nanstd(warped) + cfg.eps)
                temporal = cfg.temporal_strength * flow_conf * np.exp(-disagreement)
                temporal = np.clip(temporal, 0.0, 0.95)
                out = temporal * warped + (1.0 - temporal) * calibrated
                outputs[i] = np.where(_valid(out), out, warped).astype(np.float32)
                scales[i], offsets[i], confidences[i] = a, b, fit_conf * float(np.mean(flow_conf))
                previous_index = i

        stacked = np.stack([_from_working(x, cfg.depth_mode, cfg.eps) for x in outputs])
        return SyncResult(stacked.astype(np.float32), scales.astype(np.float32), offsets.astype(np.float32), confidences.astype(np.float32))
