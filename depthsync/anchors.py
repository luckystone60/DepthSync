"""Explicit offline photo-anchor selection for validation."""
from __future__ import annotations

from pathlib import Path

import numpy as np


ANCHOR_FILENAMES = {
    "dav2-large": "photo_dav2_large_disparity.npy",
    "depthpro": "photo_disparity.npy",
}


def selected_anchor_path(depth_dir: Path | str, anchor_model: str = "dav2-large") -> Path:
    try:
        filename = ANCHOR_FILENAMES[anchor_model]
    except KeyError as error:
        raise ValueError(f"unknown anchor model: {anchor_model}") from error
    return Path(depth_dir) / filename


def load_anchor_disparity(depth_dir: Path | str, anchor_model: str = "dav2-large") -> np.ndarray:
    path = selected_anchor_path(depth_dir, anchor_model)
    if not path.is_file():
        raise ValueError(f"selected anchor is missing: {path}")
    anchor = np.load(path).astype(np.float32)
    if anchor.ndim != 2:
        raise ValueError("selected anchor must have shape [H,W]")
    if not np.all(np.isfinite(anchor)):
        raise ValueError("selected anchor must be finite")
    return anchor
