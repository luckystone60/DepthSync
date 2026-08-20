from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .core import DepthSync, DepthSyncConfig
from .flow import load_dense_flow
from .local_alignment import (
    LocalAlignmentConfig,
    apply_local_sequence,
    fit_flow_guided_fields,
    save_local_fields,
    select_temporal_safe_fields,
)

def main() -> None:
    p = argparse.ArgumentParser(description="Align video depth to a high-quality photo depth anchor")
    p.add_argument("--video-depth", required=True, help=".npy array shaped [T,H,W]")
    p.add_argument("--photo-depth", required=True, help=".npy array shaped [H,W]")
    p.add_argument("--anchor", required=True, type=int, help="video index matching the photo")
    p.add_argument("--output", required=True, help="output .npz")
    p.add_argument("--mode", choices=("depth", "disparity"), default="depth")
    p.add_argument("--flow", type=Path, help="optional cached dense bidirectional flow NPZ")
    p.add_argument("--face-box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    args = p.parse_args()
    video = np.load(args.video_depth)
    photo = np.load(args.photo_depth)
    sync = DepthSync(DepthSyncConfig(depth_mode=args.mode))
    result = sync(list(video), photo, args.anchor)
    output_depths = result.depths
    local_fields = None
    local_strength = 0.0
    local_safety: dict[str, float] = {}
    if args.flow is not None:
        dense_flow = load_dense_flow(args.flow)
        photo_low = cv2.resize(
            photo.astype(np.float32),
            (video.shape[2], video.shape[1]),
            interpolation=cv2.INTER_AREA,
        )
        local_config = LocalAlignmentConfig()
        local_fields = fit_flow_guided_fields(
            result.depths, photo_low, args.anchor, dense_flow, local_config
        )
        local_fields, local_strength, local_safety = select_temporal_safe_fields(
            result.depths,
            local_fields,
            dense_flow,
            local_config,
            photo_anchor=photo_low if args.face_box else None,
            anchor_index=args.anchor if args.face_box else None,
            priority_box=tuple(args.face_box) if args.face_box else None,
        )
        output_depths = apply_local_sequence(result.depths, local_fields, local_config)
    np.savez_compressed(
        args.output,
        depths=output_depths,
        scales=result.scales,
        offsets=result.offsets,
        confidences=result.confidences,
        lut_x=result.lut_x,
        lut_y=result.lut_y,
        fallback_reasons=np.asarray(result.fallback_reasons),
        local_strength=np.float32(local_strength),
        **{key: np.float32(value) for key, value in local_safety.items()},
    )
    if local_fields is not None:
        field_path = str(Path(args.output).with_suffix(".local_fields.npz"))
        save_local_fields(field_path, local_fields)


if __name__ == "__main__":
    main()
