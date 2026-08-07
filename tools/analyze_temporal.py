"""Inspect frame-to-frame DepthSync changes using the validation block motion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from depthsync.core import DepthSync, DepthSyncConfig, FrameParameters
from depthsync.motion import load_motion


def robust_range(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, 0.9) - np.quantile(finite, 0.1)) + 1e-6


def warp_previous(previous: np.ndarray, field: np.ndarray) -> np.ndarray:
    height, width = previous.shape
    dense = cv2.resize(field.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    return cv2.remap(
        previous,
        xx + dense[..., 0] * max(width - 1, 1),
        yy + dense[..., 1] * max(height - 1, 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    scene = args.scene
    root = args.root
    raw = np.load(root / "artifacts/depth" / scene / "video_disparity.npz")["disparity"].astype(np.float32)
    photo = np.load(root / "artifacts/depth" / scene / "photo_disparity.npy").astype(np.float32)
    affine = np.load(root / "results" / scene / "affine_depth.npz")["disparity"].astype(np.float32)
    synced = np.load(root / "results" / scene / "synced_depth.npz")["disparity"].astype(np.float32)
    params = np.load(root / "results" / scene / "v4_parameters.npz")
    motion = load_motion(root / "results" / scene / "motion.npz")
    scale = robust_range(photo)
    sync = DepthSync(DepthSyncConfig(depth_mode="disparity"))
    base = np.stack(
        [
            sync.apply_frame(
                depth,
                FrameParameters(
                    float(params["scales"][index]),
                    float(params["offsets"][index]),
                    float(params["confidences"][index]),
                    str(params["fallback_reasons"][index]),
                    params["lut_x"][index],
                    params["lut_y"][index],
                    None,
                ),
            )
            for index, depth in enumerate(raw)
        ]
    )
    rows: list[dict[str, float | int]] = []
    for index in range(1, len(raw)):
        field = motion.to_previous[index]
        confidence = cv2.resize(
            motion.confidence_previous[index],
            (raw.shape[2], raw.shape[1]),
            interpolation=cv2.INTER_LINEAR,
        )
        valid = confidence > 0.15

        def error(sequence: np.ndarray) -> float:
            previous = warp_previous(sequence[index - 1], field)
            mask = valid & np.isfinite(previous) & np.isfinite(sequence[index])
            return float(np.median(np.abs(sequence[index][mask] - previous[mask])) / scale)

        region_delta = float(
            np.max(
                np.abs(
                    params["region_offsets"][index]
                    - params["region_offsets"][index - 1]
                )
            )
            / scale
        )
        x = np.linspace(
            float(np.quantile(raw[index], 0.02)),
            float(np.quantile(raw[index], 0.98)),
            128,
            dtype=np.float32,
        )
        previous_mapping = np.interp(x, params["lut_x"][index - 1], params["lut_y"][index - 1])
        current_mapping = np.interp(x, params["lut_x"][index], params["lut_y"][index])
        rows.append(
            {
                "frame": index,
                "raw": error(raw),
                "affine": error(affine),
                "lut_base": error(base),
                "v4": error(synced),
                "lut_delta": float(np.median(np.abs(current_mapping - previous_mapping)) / scale),
                "region_delta": region_delta,
                "confidence": float(params["confidences"][index]),
            }
        )
    for key in ("v4", "lut_base", "lut_delta", "region_delta"):
        top = sorted(rows, key=lambda row: float(row[key]), reverse=True)[:10]
        print(key, json.dumps(top, ensure_ascii=False))


if __name__ == "__main__":
    main()
