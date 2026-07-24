"""Run the official validation models through an isolated Python environment.

This script deliberately lives outside the core package: model dependencies are
large and are not part of the edge implementation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class InferenceMetadata:
    backend: str
    device: str
    precision: str
    frame_count: int
    output_height: int
    output_width: int
    elapsed_seconds: float
    checkpoint: str


def _output_size(width: int, height: int, short_side: int) -> tuple[int, int]:
    scale = short_side / min(width, height)
    return max(2, int(round(width * scale))), max(2, int(round(height * scale)))


def _write_metadata(path: Path, metadata: InferenceMetadata) -> None:
    path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")


def run_depthpro(anchor_path: Path, output_dir: Path, checkpoint: Path) -> None:
    sys.path.insert(0, str(ROOT / "third_party" / "ml-depth-pro" / "src"))
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = torch.float16 if device.type == "cuda" else torch.float32
    config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(checkpoint.resolve()))
    start = time.perf_counter()
    model, transform = depth_pro.create_model_and_transforms(config=config, device=device, precision=precision)
    model.eval()
    bgr = cv2.imread(str(anchor_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read anchor image: {anchor_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prediction = model.infer(transform(rgb))
    depth = prediction["depth"].detach().float().cpu().numpy()
    disparity = 1.0 / np.maximum(depth, 1e-6)
    elapsed = time.perf_counter() - start
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "photo_depth.npy", depth.astype(np.float32))
    np.save(output_dir / "photo_disparity.npy", disparity.astype(np.float32))
    _write_metadata(
        output_dir / "depthpro.json",
        InferenceMetadata("DepthPro", str(device), str(precision), 1, depth.shape[0], depth.shape[1], elapsed, str(checkpoint)),
    )


def run_vda_streaming(clip_path: Path, output_dir: Path, checkpoint: Path, short_side: int, input_size: int) -> None:
    repo = ROOT / "third_party" / "Video-Depth-Anything"
    sys.path.insert(0, str(repo))
    from video_depth_anything.video_depth_stream import VideoDepthAnything

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot read validation clip: {clip_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width, out_height = _output_size(width, height, short_side)
    depths: list[np.ndarray] = []
    start = time.perf_counter()
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = model.infer_video_depth_one(rgb, input_size=input_size, device=device, fp32=device == "cpu")
            depths.append(cv2.resize(depth.astype(np.float32), (out_width, out_height), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
    elapsed = time.perf_counter() - start
    if len(depths) != 90:
        raise ValueError(f"Expected 90 VDA outputs, got {len(depths)} for {clip_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stacked = np.stack(depths).astype(np.float32)
    np.savez_compressed(output_dir / "video_disparity.npz", disparity=stacked)
    _write_metadata(
        output_dir / "vda_streaming.json",
        InferenceMetadata("Video-Depth-Anything-Small-stream", device, "fp16" if device == "cuda" else "fp32", len(depths), out_height, out_width, elapsed, str(checkpoint)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="backend", required=True)
    depthpro = subparsers.add_parser("depthpro")
    depthpro.add_argument("--anchor", required=True, type=Path)
    depthpro.add_argument("--output-dir", required=True, type=Path)
    depthpro.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "depth_pro.pt")
    vda = subparsers.add_parser("vda")
    vda.add_argument("--clip", required=True, type=Path)
    vda.add_argument("--output-dir", required=True, type=Path)
    vda.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "video_depth_anything_vits.pth")
    vda.add_argument("--short-side", type=int, default=256)
    vda.add_argument("--input-size", type=int, default=518)
    args = parser.parse_args()
    if args.backend == "depthpro":
        run_depthpro(args.anchor, args.output_dir, args.checkpoint)
    else:
        run_vda_streaming(args.clip, args.output_dir, args.checkpoint, args.short_side, args.input_size)


if __name__ == "__main__":
    main()
