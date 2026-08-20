"""Render depth maps and videos with explicit, reproducible value ranges."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .anchors import load_anchor_disparity


def robust_limits(values: np.ndarray, lower: float = 0.02, upper: float = 0.98) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.quantile(finite, (lower, upper))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def normalize_depth(depth: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    low, high = limits
    normalized = (np.nan_to_num(depth, nan=low, posinf=high, neginf=low) - low) / max(high - low, 1e-12)
    return np.clip(normalized, 0.0, 1.0)


def colorize_depth(depth: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    normalized = normalize_depth(depth, limits)
    # Ordered sub-LSB dither prevents false contouring on large smooth planes
    # after 8-bit Turbo quantization and MP4 encoding. The pattern is fixed, so
    # it does not introduce temporal noise into the comparison video.
    bayer4 = np.asarray(
        [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
        np.float32,
    )
    dither = (bayer4 - 7.5) / (16.0 * 255.0)
    tiled = np.tile(
        dither,
        ((normalized.shape[0] + 3) // 4, (normalized.shape[1] + 3) // 4),
    )[: normalized.shape[0], : normalized.shape[1]]
    gray = (np.clip(normalized + tiled, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def depth_to_gray16(depth: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    return (normalize_depth(depth, limits) * 65535.0).round().astype(np.uint16)


def _colorbar(width: int, height: int, limits: tuple[float, float], color: bool) -> np.ndarray:
    gradient = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    gradient = np.repeat(gradient, height, axis=0)
    bar = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO) if color else cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    low, high = limits
    cv2.putText(bar, f"far {low:.4g}", (4, height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    high_text = f"near {high:.4g}"
    text_width = cv2.getTextSize(high_text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
    cv2.putText(bar, high_text, (max(4, width - text_width - 4), height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def _panel(image: np.ndarray, label: str, size: tuple[int, int], limits: tuple[float, float] | None, colorbar: bool = True) -> np.ndarray:
    content = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    header = np.zeros((28, size[0], 3), np.uint8)
    cv2.putText(header, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    parts = [header, content]
    if colorbar and limits is not None:
        parts.append(_colorbar(size[0], 20, limits, True))
    elif colorbar:
        parts.append(np.zeros((20, size[0], 3), np.uint8))
    return np.vstack(parts)


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames, fps


def _write_depth_video(path: Path, depths: np.ndarray, limits: tuple[float, float], fps: float, color: bool) -> None:
    content_width = int(round(depths.shape[2] / 2.0) * 2)
    content_height = int(round(depths.shape[1] / 2.0) * 2)
    frame_size = (content_width, content_height + 48)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if not writer.isOpened():
        raise ValueError(f"Cannot create video: {path}")
    label = "disparity (near = high)"
    for depth in depths:
        if color:
            visual = colorize_depth(depth, limits)
        else:
            gray = (normalize_depth(depth, limits) * 255.0).round().astype(np.uint8)
            visual = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        visual = cv2.resize(visual, (content_width, content_height), interpolation=cv2.INTER_LINEAR)
        header = np.zeros((28, content_width, 3), np.uint8)
        cv2.putText(header, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(np.vstack((header, visual, _colorbar(content_width, 20, limits, color))))
    writer.release()


def render_scene(
    scene: str,
    clip_root: Path,
    depth_root: Path,
    result_root: Path,
    anchor_model: str = "dav2-large",
) -> dict[str, object]:
    clip_dir = clip_root / scene
    depth_dir = depth_root / scene
    output_dir = result_root / scene / "depth_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(depth_dir / "video_disparity.npz")["disparity"].astype(np.float32)
    v4 = np.load(result_root / scene / "v4_depth.npz")["disparity"].astype(np.float32)
    local = np.load(result_root / scene / "v41_local_depth.npz")["disparity"].astype(np.float32)
    photo = load_anchor_disparity(depth_dir, anchor_model)
    manifest = json.loads((clip_dir / "manifest.json").read_text(encoding="utf-8"))
    anchor = int(manifest["anchor_index"])
    fps = float(manifest["target_fps"])
    if len(raw) != len(v4) or len(raw) != len(local):
        raise ValueError(f"Frame count mismatch for scene {scene}")
    photo_low = cv2.resize(photo, (raw.shape[2], raw.shape[1]), interpolation=cv2.INTER_AREA)
    photo_limits = robust_limits(photo_low)
    raw_limits = robust_limits(raw)
    _write_depth_video(output_dir / "v4_depth_color.mp4", v4, photo_limits, fps, True)
    _write_depth_video(output_dir / "v4_depth_gray.mp4", v4, photo_limits, fps, False)
    _write_depth_video(output_dir / "v41_local_depth_color.mp4", local, photo_limits, fps, True)
    _write_depth_video(output_dir / "v41_local_depth_gray.mp4", local, photo_limits, fps, False)
    panel_size = (320, 180)
    photo_color = colorize_depth(photo_low, photo_limits)
    comparison_size = (panel_size[0] * 4, panel_size[1] + 48)
    comparison_name = "depth_comparison_dav2_v41_local.mp4" if anchor_model == "dav2-large" else "depth_comparison_depthpro_v41_local.mp4"
    comparison_path = output_dir / comparison_name
    writer = cv2.VideoWriter(str(comparison_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, comparison_size)
    if not writer.isOpened():
        raise ValueError(f"Cannot create video: {comparison_path}")
    for index in range(len(raw)):
        panels = [
            _panel(colorize_depth(raw[index], raw_limits), f"VDA normalized / frame {index:02d}", panel_size, raw_limits),
            _panel(colorize_depth(v4[index], photo_limits), "V4.1 adaptive global LUT", panel_size, photo_limits),
            _panel(colorize_depth(local[index], photo_limits), "V4.1 + flow local field", panel_size, photo_limits),
            _panel(
                photo_color,
                "DAv2-Large photo anchor" if anchor_model == "dav2-large" else "DepthPro photo anchor",
                panel_size,
                photo_limits,
            ),
        ]
        writer.write(np.hstack(panels))
    writer.release()
    cv2.imwrite(str(output_dir / "anchor_v4_gray16.png"), depth_to_gray16(v4[anchor], photo_limits))
    cv2.imwrite(str(output_dir / f"anchor_{anchor_model}_gray16.png"), depth_to_gray16(photo_low, photo_limits))
    if anchor_model == "dav2-large":
        depthpro_path = depth_dir / "photo_disparity.npy"
        if depthpro_path.is_file():
            depthpro = load_anchor_disparity(depth_dir, "depthpro")
            depthpro_low = cv2.resize(depthpro, (raw.shape[2], raw.shape[1]), interpolation=cv2.INTER_AREA)
            depthpro_color = colorize_depth(depthpro_low, photo_limits)
            comparison = np.hstack(
                [
                    _panel(photo_color, "DAv2-Large anchor / fixed DAv2 range", panel_size, photo_limits),
                    _panel(depthpro_color, "DepthPro anchor / fixed DAv2 range", panel_size, photo_limits),
                ]
            )
            cv2.imwrite(str(output_dir / "dav2_vs_depthpro_anchor.png"), comparison)

    metadata = {
        "scene": scene,
        "anchor_model": anchor_model,
        "frame_count": len(raw),
        "fps": fps,
        "anchor_index": anchor,
        "shared_limits_p02_p98": photo_limits,
        "vda_sequence_limits_p02_p98": raw_limits,
        "files": {
            "comparison_video": comparison_name,
            "v4_color_video": "v4_depth_color.mp4",
            "v4_gray_video": "v4_depth_gray.mp4",
            "local_color_video": "v41_local_depth_color.mp4",
            "local_gray_video": "v41_local_depth_gray.mp4",
        },
    }
    (output_dir / "visualization.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Render comparable depth maps and videos")
    parser.add_argument("--clip-root", type=Path, default=Path("artifacts/clips"))
    parser.add_argument("--depth-root", type=Path, default=Path("artifacts/depth"))
    parser.add_argument("--result-root", type=Path, default=Path("results"))
    parser.add_argument("--compare-dir", type=Path, default=None)
    parser.add_argument("--anchor-model", choices=("dav2-large", "depthpro"), default="dav2-large")
    parser.add_argument("--scenes", nargs="+", default=["01", "02", "03"])
    args = parser.parse_args()
    for scene in args.scenes:
        metadata = render_scene(
            scene,
            args.clip_root,
            args.depth_root,
            args.result_root,
            args.anchor_model,
        )
        compare_dir = args.compare_dir or (args.result_root / "compare_videos")
        compare_dir.mkdir(parents=True, exist_ok=True)
        source = args.result_root / scene / "depth_visualization" / metadata["files"]["comparison_video"]
        shutil.copy2(source, compare_dir / f"{scene}_{source.name}")
        print(f"{scene}: {metadata['frame_count']} frames -> {args.result_root / scene / 'depth_visualization'}")


if __name__ == "__main__":
    main()
