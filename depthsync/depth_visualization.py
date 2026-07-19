"""Render depth maps and videos with explicit, reproducible value ranges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


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
    gray = (normalize_depth(depth, limits) * 255.0).round().astype(np.uint8)
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


def render_scene(scene: str, clip_root: Path, depth_root: Path, result_root: Path) -> dict[str, object]:
    clip_dir = clip_root / scene
    depth_dir = depth_root / scene
    output_dir = result_root / scene / "depth_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_frames, fps = _read_video(clip_dir / "clip.mp4")
    raw = np.load(depth_dir / "video_disparity.npz")["disparity"].astype(np.float32)
    synced = np.load(result_root / scene / "synced_depth.npz")["disparity"].astype(np.float32)
    photo = np.load(depth_dir / "photo_disparity.npy").astype(np.float32)
    manifest = json.loads((clip_dir / "manifest.json").read_text(encoding="utf-8"))
    anchor = int(manifest["anchor_index"])
    if len(rgb_frames) != len(raw) or len(raw) != len(synced):
        raise ValueError(f"Frame count mismatch for scene {scene}")

    photo_low = cv2.resize(photo, (raw.shape[2], raw.shape[1]), interpolation=cv2.INTER_AREA)
    raw_limits = robust_limits(raw)
    photo_limits = robust_limits(photo_low)
    synced_limits = photo_limits

    _write_depth_video(output_dir / "vda_raw_color.mp4", raw, raw_limits, fps, True)
    _write_depth_video(output_dir / "vda_raw_gray.mp4", raw, raw_limits, fps, False)
    _write_depth_video(output_dir / "depthsync_color.mp4", synced, synced_limits, fps, True)
    _write_depth_video(output_dir / "depthsync_gray.mp4", synced, synced_limits, fps, False)

    panel_size = (320, 180)
    photo_color = colorize_depth(photo_low, photo_limits)
    comparison_size = (panel_size[0] * 5, panel_size[1] + 48)
    comparison_path = output_dir / "depth_comparison.mp4"
    writer = cv2.VideoWriter(str(comparison_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, comparison_size)
    if not writer.isOpened():
        raise ValueError(f"Cannot create video: {comparison_path}")
    anchor_comparison = None
    for index, rgb in enumerate(rgb_frames):
        panels = [
            _panel(rgb, f"RGB frame {index:02d}", panel_size, None),
            _panel(colorize_depth(raw[index], raw_limits), "VDA raw / own range", panel_size, raw_limits),
            _panel(colorize_depth(raw[index], photo_limits), "VDA raw / photo range", panel_size, photo_limits),
            _panel(colorize_depth(synced[index], synced_limits), "DepthSync / photo range", panel_size, synced_limits),
            _panel(photo_color, "DepthPro anchor / photo range", panel_size, photo_limits),
        ]
        frame = np.hstack(panels)
        writer.write(frame)
        if index == anchor:
            anchor_comparison = frame
    writer.release()
    if anchor_comparison is None:
        raise ValueError(f"Anchor frame {anchor} missing for scene {scene}")

    anchor_raw = colorize_depth(raw[anchor], raw_limits)
    anchor_synced = colorize_depth(synced[anchor], photo_limits)
    cv2.imwrite(str(output_dir / "anchor_comparison.png"), anchor_comparison)
    cv2.imwrite(str(output_dir / "anchor_vda_raw_color.png"), anchor_raw)
    cv2.imwrite(str(output_dir / "anchor_depthsync_color.png"), anchor_synced)
    cv2.imwrite(str(output_dir / "anchor_depthpro_color.png"), photo_color)
    cv2.imwrite(str(output_dir / "anchor_depthsync_gray16.png"), depth_to_gray16(synced[anchor], photo_limits))
    cv2.imwrite(str(output_dir / "anchor_depthpro_gray16.png"), depth_to_gray16(photo_low, photo_limits))

    metadata: dict[str, object] = {
        "scene": scene,
        "frame_count": len(raw),
        "fps": fps,
        "anchor_index": anchor,
        "disparity_direction": "larger values are nearer",
        "raw_limits_p02_p98": raw_limits,
        "photo_and_synced_limits_p02_p98": photo_limits,
        "files": {
            "comparison_video": "depth_comparison.mp4",
            "raw_color_video": "vda_raw_color.mp4",
            "raw_gray_video": "vda_raw_gray.mp4",
            "synced_color_video": "depthsync_color.mp4",
            "synced_gray_video": "depthsync_gray.mp4",
            "anchor_comparison": "anchor_comparison.png",
        },
    }
    (output_dir / "visualization.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Render comparable depth maps and videos")
    parser.add_argument("--clip-root", type=Path, default=Path("artifacts/clips"))
    parser.add_argument("--depth-root", type=Path, default=Path("artifacts/depth"))
    parser.add_argument("--result-root", type=Path, default=Path("results"))
    parser.add_argument("--scenes", nargs="+", default=["01", "02", "03"])
    args = parser.parse_args()
    for scene in args.scenes:
        metadata = render_scene(scene, args.clip_root, args.depth_root, args.result_root)
        print(f"{scene}: {metadata['frame_count']} frames -> {args.result_root / scene / 'depth_visualization'}")


if __name__ == "__main__":
    main()
