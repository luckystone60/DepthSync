from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .core import DepthSync, DepthSyncConfig


def _load_rgb(path: Path) -> list[np.ndarray]:
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    frames = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in files]
    if any(x is None for x in frames):
        raise ValueError(f"Failed to read an RGB frame from {path}")
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description="Align video depth to a high-quality photo depth anchor")
    p.add_argument("--video-depth", required=True, help=".npy array shaped [T,H,W]")
    p.add_argument("--photo-depth", required=True, help=".npy array shaped [H,W]")
    p.add_argument("--anchor", required=True, type=int, help="video index matching the photo")
    p.add_argument("--rgb-dir", help="optional sorted RGB frames for optical-flow propagation")
    p.add_argument("--output", required=True, help="output .npz")
    p.add_argument("--mode", choices=("depth", "disparity"), default="depth")
    p.add_argument("--temporal-strength", type=float, default=0.72)
    p.add_argument("--detail-strength", type=float, default=0.35)
    args = p.parse_args()
    video = np.load(args.video_depth)
    photo = np.load(args.photo_depth)
    rgb = _load_rgb(Path(args.rgb_dir)) if args.rgb_dir else None
    sync = DepthSync(DepthSyncConfig(depth_mode=args.mode, temporal_strength=args.temporal_strength, detail_strength=args.detail_strength))
    result = sync(list(video), photo, args.anchor, rgb)
    np.savez_compressed(
        args.output,
        depths=result.depths,
        scales=result.scales,
        offsets=result.offsets,
        confidences=result.confidences,
        fallback_reasons=np.asarray(result.fallback_reasons),
    )


if __name__ == "__main__":
    main()
