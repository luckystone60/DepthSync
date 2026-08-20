"""Conservative photo-to-video anchor registration for offline preparation."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PhotoRegistration:
    """Normalized video-pixel to photo-pixel sampling grid."""

    grid: np.ndarray
    confidence: float
    method: str
    identity_error: float
    registered_error: float


def _gray(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 3:
        values = cv2.cvtColor(values, cv2.COLOR_BGR2GRAY)
    if values.ndim != 2:
        raise ValueError("registration images must be grayscale or BGR")
    return cv2.resize(values.astype(np.float32), (shape[1], shape[0]), cv2.INTER_AREA)


def identity_grid(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    return np.stack((xx / max(width - 1, 1), yy / max(height - 1, 1)), axis=-1)


def estimate_photo_registration(
    photo_rgb: np.ndarray,
    video_anchor_rgb: np.ndarray,
    output_shape: tuple[int, int],
    max_side: int = 640,
    minimum_improvement: float = 0.08,
) -> PhotoRegistration:
    """Estimate a guarded affine registration and otherwise return identity.

    The synthetic validation anchor is usually the same decoded frame at a
    different resolution.  In that case identity is intentionally preserved;
    ECC is accepted only when it materially lowers robust luminance error and
    remains close to a camera crop/scale transform.
    """
    out_h, out_w = output_shape
    scale = min(1.0, max_side / max(out_h, out_w))
    work_shape = (max(32, int(round(out_h * scale))), max(32, int(round(out_w * scale))))
    photo = _gray(photo_rgb, work_shape)
    video = _gray(video_anchor_rgb, work_shape)
    photo = cv2.GaussianBlur(photo, (5, 5), 0)
    video = cv2.GaussianBlur(video, (5, 5), 0)
    identity_error = float(np.median(np.abs(photo - video)))
    identity = PhotoRegistration(
        identity_grid(output_shape), 1.0, "identity", identity_error, identity_error
    )
    # Compression-only differences do not justify a geometric warp.
    if identity_error <= 4.0:
        return identity

    warp = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    try:
        _, warp = cv2.findTransformECC(
            video,
            photo,
            warp,
            cv2.MOTION_AFFINE,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5),
            inputMask=None,
            gaussFiltSize=5,
        )
    except cv2.error:
        return identity
    linear = warp[:, :2].astype(np.float64)
    singular = np.linalg.svd(linear, compute_uv=False)
    translation = np.abs(warp[:, 2]) / np.asarray([work_shape[1], work_shape[0]])
    if np.any(singular < 0.90) or np.any(singular > 1.10) or np.any(translation > 0.08):
        return identity
    aligned = cv2.warpAffine(
        photo,
        warp,
        (work_shape[1], work_shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT101,
    )
    registered_error = float(np.median(np.abs(aligned - video)))
    improvement = (identity_error - registered_error) / max(identity_error, 1e-6)
    if improvement < minimum_improvement:
        return identity

    yy, xx = np.mgrid[:out_h, :out_w].astype(np.float32)
    work_x = xx * (work_shape[1] - 1) / max(out_w - 1, 1)
    work_y = yy * (work_shape[0] - 1) / max(out_h - 1, 1)
    map_x = warp[0, 0] * work_x + warp[0, 1] * work_y + warp[0, 2]
    map_y = warp[1, 0] * work_x + warp[1, 1] * work_y + warp[1, 2]
    grid = np.stack(
        (map_x / max(work_shape[1] - 1, 1), map_y / max(work_shape[0] - 1, 1)),
        axis=-1,
    ).astype(np.float32)
    confidence = float(np.clip(improvement / 0.30, 0.0, 1.0))
    return PhotoRegistration(grid, confidence, "ecc_affine", identity_error, registered_error)


def remap_photo_depth(
    photo_depth: np.ndarray,
    registration: PhotoRegistration,
    output_shape: tuple[int, int],
) -> np.ndarray:
    grid = registration.grid
    if grid.shape[:2] != output_shape:
        grid = cv2.resize(grid, (output_shape[1], output_shape[0]), cv2.INTER_LINEAR)
    map_x = grid[..., 0] * max(photo_depth.shape[1] - 1, 1)
    map_y = grid[..., 1] * max(photo_depth.shape[0] - 1, 1)
    return cv2.remap(
        photo_depth.astype(np.float32),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
