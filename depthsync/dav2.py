"""Pure contracts for offline Depth Anything V2 anchor outputs."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def orient_relative_depth(
    relative: np.ndarray,
    reference_disparity: np.ndarray,
    minimum_correlation: float = 0.05,
) -> tuple[np.ndarray, bool]:
    """Orient a relative-depth map to increase with reference disparity."""
    value = np.asarray(relative, dtype=np.float32)
    reference = np.asarray(reference_disparity, dtype=np.float32)
    if value.ndim != 2 or reference.ndim != 2:
        raise ValueError("relative depth and reference disparity must be two-dimensional")
    if not np.all(np.isfinite(value)) or not np.all(np.isfinite(reference)):
        raise ValueError("relative depth and reference disparity must be finite")
    if reference.shape != value.shape:
        reference = cv2.resize(
            reference,
            (value.shape[1], value.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    valid = np.isfinite(value) & np.isfinite(reference)
    if int(np.count_nonzero(valid)) < 64:
        raise ValueError("insufficient paired samples for DAv2 direction")
    source = value[valid].astype(np.float64)
    target = reference[valid].astype(np.float64)
    if float(np.std(source)) <= 1e-8 or float(np.std(target)) <= 1e-8:
        raise ValueError("ambiguous DAv2 direction due to constant input")
    correlation = float(np.corrcoef(source, target)[0, 1])
    if not np.isfinite(correlation) or abs(correlation) < minimum_correlation:
        raise ValueError("ambiguous DAv2 direction")
    if correlation >= 0.0:
        return value, False
    minimum = float(np.min(value))
    maximum = float(np.max(value))
    return (minimum + maximum - value).astype(np.float32), True


def write_dav2_outputs(
    output_dir: Path | str,
    relative: np.ndarray,
    disparity: np.ndarray,
    metadata: dict[str, object],
) -> None:
    """Persist the offline DAv2 anchor contract without model dependencies."""
    relative = np.asarray(relative, dtype=np.float32)
    disparity = np.asarray(disparity, dtype=np.float32)
    if relative.ndim != 2 or disparity.shape != relative.shape:
        raise ValueError("DAv2 relative and disparity outputs must have equal [H,W] shapes")
    if not np.all(np.isfinite(relative)) or not np.all(np.isfinite(disparity)):
        raise ValueError("DAv2 outputs must be finite")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "photo_dav2_large_relative.npy", relative)
    np.save(destination / "photo_dav2_large_disparity.npy", disparity)
    (destination / "dav2_large.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
