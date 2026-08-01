"""Render depth maps and videos with explicit, reproducible value ranges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def select_inspection_frames(
    scene: str,
    frame_count: int,
    anchor_index: int,
    worst_jump_frame: int,
    configured: list[int] | None,
) -> list[int]:
    """Return deterministic, valid frames for subjective V5 inspection."""
    if configured is not None:
        requested = configured
    elif scene == "02":
        requested = list(range(75, 90))
    elif scene == "03":
        requested = [anchor_index, worst_jump_frame - 1, worst_jump_frame, worst_jump_frame + 1]
    else:
        requested = [0, anchor_index, 60]
    return sorted({min(max(int(index), 0), frame_count - 1) for index in requested})


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
    flow_root: Path = Path("artifacts/flow"),
    validation_config: Path = Path("config/validation-scenes.json"),
) -> dict[str, object]:
    clip_dir = clip_root / scene
    depth_dir = depth_root / scene
    output_dir = result_root / scene / "depth_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(depth_dir / "video_disparity.npz")["disparity"].astype(np.float32)
    v4 = np.load(result_root / scene / "v4_depth.npz")["disparity"].astype(np.float32)
    v5 = np.load(result_root / scene / "v5_depth.npz")["disparity"].astype(np.float32)
    fields = np.load(result_root / scene / "v5_parameters.npz")
    delta_scale = fields["delta_scale"].astype(np.float32)
    offset_norm = fields["offset_norm"].astype(np.float32)
    field_confidence = fields["confidence"].astype(np.float32)
    flow = np.load(flow_root / scene / "sea_raft_s_flow.npz")
    photo = np.load(depth_dir / "photo_disparity.npy").astype(np.float32)
    manifest = json.loads((clip_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((result_root / scene / "metrics.json").read_text(encoding="utf-8"))
    anchor = int(manifest["anchor_index"])
    jump_frame = int(metrics["v5_excess_jump_frame"])
    fps = float(manifest["target_fps"])
    if len(raw) != len(v4) or len(raw) != len(v5):
        raise ValueError(f"Frame count mismatch for scene {scene}")

    photo_low = cv2.resize(photo, (raw.shape[2], raw.shape[1]), interpolation=cv2.INTER_AREA)
    photo_limits = robust_limits(photo_low)
    _write_depth_video(output_dir / "v5_depth_color.mp4", v5, photo_limits, fps, True)
    _write_depth_video(output_dir / "v5_depth_gray.mp4", v5, photo_limits, fps, False)

    panel_size = (320, 180)
    photo_color = colorize_depth(photo_low, photo_limits)
    comparison_size = (panel_size[0] * 4, panel_size[1] + 48)
    comparison_path = output_dir / "depth_comparison_v5.mp4"
    writer = cv2.VideoWriter(str(comparison_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, comparison_size)
    if not writer.isOpened():
        raise ValueError(f"Cannot create video: {comparison_path}")
    diagnostic_path = output_dir / "field_diagnostics_v5.mp4"
    diagnostic_writer = cv2.VideoWriter(
        str(diagnostic_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, comparison_size
    )
    if not diagnostic_writer.isOpened():
        raise ValueError(f"Cannot create video: {diagnostic_path}")
    flow_confidence = np.ones_like(field_confidence)
    backward_confidence = flow["confidence_previous"].astype(np.float32)
    for index in range(1, len(flow_confidence)):
        flow_confidence[index] = cv2.resize(
            backward_confidence[index - 1],
            (field_confidence.shape[2], field_confidence.shape[1]),
            interpolation=cv2.INTER_AREA,
        )
    scene_config = {}
    if validation_config.exists():
        scene_config = json.loads(validation_config.read_text(encoding="utf-8")).get(scene, {})
    inspection = select_inspection_frames(
        scene,
        len(raw),
        anchor,
        jump_frame,
        scene_config.get("inspection_frames"),
    )
    key_frames: dict[int, np.ndarray] = {}
    for index in range(len(raw)):
        panels = [
            _panel(colorize_depth(raw[index], photo_limits), f"VDA raw / frame {index:02d}", panel_size, photo_limits),
            _panel(colorize_depth(v4[index], photo_limits), "V4 global monotonic LUT", panel_size, photo_limits),
            _panel(colorize_depth(v5[index], photo_limits), "V5 flow local affine field", panel_size, photo_limits),
            _panel(photo_color, "DepthPro photo anchor", panel_size, photo_limits),
        ]
        frame = np.hstack(panels)
        writer.write(frame)
        if index in inspection:
            key_frames[index] = frame.copy()
        diagnostics = [
            _panel(colorize_depth(delta_scale[index], (-0.2, 0.2)), "delta scale [-0.2, 0.2]", panel_size, (-0.2, 0.2)),
            _panel(colorize_depth(offset_norm[index], (-0.2, 0.2)), "normalized offset [-0.2, 0.2]", panel_size, (-0.2, 0.2)),
            _panel(colorize_depth(field_confidence[index], (0.0, 1.0)), "field confidence [0, 1]", panel_size, (0.0, 1.0)),
            _panel(colorize_depth(flow_confidence[index], (0.0, 1.0)), "flow consistency [0, 1]", panel_size, (0.0, 1.0)),
        ]
        diagnostic_writer.write(np.hstack(diagnostics))
    writer.release()
    diagnostic_writer.release()
    for index, frame in key_frames.items():
        cv2.imwrite(str(output_dir / f"frame_{index:03d}_comparison.png"), frame)
    cv2.imwrite(str(output_dir / "anchor_v5_gray16.png"), depth_to_gray16(v5[anchor], photo_limits))
    cv2.imwrite(str(output_dir / "anchor_depthpro_gray16.png"), depth_to_gray16(photo_low, photo_limits))

    metadata: dict[str, object] = {
        "scene": scene,
        "frame_count": len(raw),
        "fps": fps,
        "anchor_index": anchor,
        "disparity_direction": "larger values are nearer",
        "shared_photo_v4_v5_limits_p02_p98": photo_limits,
        "inspection_frames": inspection,
        "files": {
            "comparison_video": "depth_comparison_v5.mp4",
            "diagnostic_video": "field_diagnostics_v5.mp4",
            "v5_color_video": "v5_depth_color.mp4",
            "v5_gray_video": "v5_depth_gray.mp4",
            "inspection_pngs": [f"frame_{index:03d}_comparison.png" for index in inspection],
        },
    }
    (output_dir / "visualization.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Render comparable depth maps and videos")
    parser.add_argument("--clip-root", type=Path, default=Path("artifacts/clips"))
    parser.add_argument("--depth-root", type=Path, default=Path("artifacts/depth"))
    parser.add_argument("--result-root", type=Path, default=Path("results"))
    parser.add_argument("--flow-root", type=Path, default=Path("artifacts/flow"))
    parser.add_argument("--validation-config", type=Path, default=Path("config/validation-scenes.json"))
    parser.add_argument("--scenes", nargs="+", default=["01", "02", "03"])
    args = parser.parse_args()
    for scene in args.scenes:
        metadata = render_scene(
            scene,
            args.clip_root,
            args.depth_root,
            args.result_root,
            args.flow_root,
            args.validation_config,
        )
        print(f"{scene}: {metadata['frame_count']} frames -> {args.result_root / scene / 'depth_visualization'}")


if __name__ == "__main__":
    main()
