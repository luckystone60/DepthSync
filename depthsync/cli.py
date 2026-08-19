from __future__ import annotations

import argparse

import numpy as np

from .core import DepthSync, DepthSyncConfig

def main() -> None:
    p = argparse.ArgumentParser(description="Align video depth to a high-quality photo depth anchor")
    p.add_argument("--video-depth", required=True, help=".npy array shaped [T,H,W]")
    p.add_argument("--photo-depth", required=True, help=".npy array shaped [H,W]")
    p.add_argument("--anchor", required=True, type=int, help="video index matching the photo")
    p.add_argument("--output", required=True, help="output .npz")
    p.add_argument("--mode", choices=("depth", "disparity"), default="depth")
    args = p.parse_args()
    video = np.load(args.video_depth)
    photo = np.load(args.photo_depth)
    sync = DepthSync(DepthSyncConfig(depth_mode=args.mode))
    result = sync(list(video), photo, args.anchor)
    np.savez_compressed(
        args.output,
        depths=result.depths,
        scales=result.scales,
        offsets=result.offsets,
        confidences=result.confidences,
        lut_x=result.lut_x,
        lut_y=result.lut_y,
        fallback_reasons=np.asarray(result.fallback_reasons),
    )


if __name__ == "__main__":
    main()
