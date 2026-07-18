"""Offline metrics and visualizations for local Live Photo simulations."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .core import DepthSync, DepthSyncConfig, MotionSequence
from .motion import estimate_block_motion, load_motion, save_motion


def _region_mask(shape: tuple[int, int], box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[:h, :w]
    return (xx >= x0 * w) & (xx < x1 * w) & (yy >= y0 * h) & (yy < y1 * h)


def _robust_range(depth: np.ndarray) -> float:
    valid = depth[np.isfinite(depth)]
    return float(np.quantile(valid, 0.9) - np.quantile(valid, 0.1)) if valid.size else 1.0


def _aligned_switch_error(
    sequence: np.ndarray,
    photo: np.ndarray,
    anchor: int,
    motion: MotionSequence,
    face_box: tuple[float, float, float, float],
    face_only: bool,
    coc: bool = False,
) -> float:
    """Compare the two anchor neighbors to MV-warped photo depth."""
    height, width = photo.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    scale = _robust_range(photo) + 1e-6
    anchor_face = _region_mask(photo.shape, face_box)
    focus = float(np.nanmedian(photo[anchor_face]))
    errors: list[float] = []
    for index, field, confidence in (
        (anchor - 1, motion.to_next[anchor - 1], motion.confidence_next[anchor - 1]),
        (anchor + 1, motion.to_previous[anchor + 1], motion.confidence_previous[anchor + 1]),
    ):
        dense = cv2.resize(field, (width, height), interpolation=cv2.INTER_LINEAR)
        map_x = xx + dense[..., 0] * max(width - 1, 1)
        map_y = yy + dense[..., 1] * max(height - 1, 1)
        reference = cv2.remap(photo, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        conf = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
        valid = np.isfinite(reference) & np.isfinite(sequence[index]) & (conf > 0.15)
        if face_only:
            x0, y0, x1, y1 = face_box
            valid &= (map_x >= x0 * width) & (map_x < x1 * width) & (map_y >= y0 * height) & (map_y < y1 * height)
        if coc:
            difference = np.abs(np.abs(sequence[index] - focus) - np.abs(reference - focus)) / scale
        else:
            difference = np.abs(sequence[index] - reference) / scale
        errors.append(float(np.nanmedian(difference[valid])) if np.any(valid) else float("nan"))
    return float(np.nanmean(errors))


def _read_frames(video: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def _depth_color(depth: np.ndarray, low: float, high: float, size: tuple[int, int]) -> np.ndarray:
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.resize(color, size, interpolation=cv2.INTER_LINEAR)


def _bokeh(rgb: np.ndarray, depth: np.ndarray, focus: float, depth_range: float, size: tuple[int, int]) -> np.ndarray:
    frame = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    d = cv2.resize(depth, size, interpolation=cv2.INTER_LINEAR)
    d = np.nan_to_num(d, nan=focus, posinf=focus, neginf=focus)
    alpha = np.clip(np.abs(d - focus) / max(depth_range * 0.65, 1e-6), 0.0, 1.0)[..., None]
    blurred = cv2.GaussianBlur(frame, (0, 0), 7.0)
    return np.clip((1.0 - alpha) * frame + alpha * blurred, 0, 255).astype(np.uint8)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def _render_comparison(
    clip: Path,
    output: Path,
    fixed: np.ndarray,
    synced: np.ndarray,
    photo: np.ndarray,
    anchor: int,
    face_mask: np.ndarray,
) -> None:
    frames = _read_frames(clip)
    if len(frames) != len(synced):
        raise ValueError("RGB/depth frame count mismatch")
    panel_size = (426, 240)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (panel_size[0] * 3, panel_size[1]))
    focus = float(np.nanmedian(photo[face_mask]))
    depth_range = _robust_range(photo)
    for index, rgb in enumerate(frames):
        fixed_depth = photo if index == anchor else fixed[index]
        synced_depth = photo if index == anchor else synced[index]
        original = _label(cv2.resize(rgb, panel_size, interpolation=cv2.INTER_AREA), f"RGB  frame {index:02d}")
        raw_bokeh = _label(_bokeh(rgb, fixed_depth, focus, depth_range, panel_size), "Fixed anchor mapping")
        synced_bokeh = _label(_bokeh(rgb, synced_depth, focus, depth_range, panel_size), "DepthSync parameter propagation")
        if index == anchor:
            cv2.putText(raw_bokeh, "PHOTO DEPTH", (130, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(synced_bokeh, "PHOTO DEPTH", (130, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        writer.write(np.hstack((original, raw_bokeh, synced_bokeh)))
    writer.release()


def evaluate_scene(scene: str, clip_root: Path, depth_root: Path, result_root: Path, face_box: tuple[float, float, float, float]) -> dict[str, float | int | str]:
    clip_dir = clip_root / scene
    depth_dir = depth_root / scene
    result_dir = result_root / scene
    result_dir.mkdir(parents=True, exist_ok=True)
    video_depth = np.load(depth_dir / "video_disparity.npz")["disparity"].astype(np.float32)
    photo = np.load(depth_dir / "photo_disparity.npy").astype(np.float32)
    anchor = int(json.loads((clip_dir / "manifest.json").read_text(encoding="utf-8"))["anchor_index"])
    photo_low = cv2.resize(photo, (video_depth.shape[2], video_depth.shape[1]), interpolation=cv2.INTER_AREA)
    motion_path = result_dir / "motion.npz"
    if motion_path.exists():
        motion = load_motion(motion_path)
    else:
        motion = estimate_block_motion(clip_dir / "clip.mp4")
        save_motion(motion_path, motion)
    sync = DepthSync(DepthSyncConfig(depth_mode="disparity"))
    start = time.perf_counter()
    result = sync.offline_prepare(video_depth, photo, anchor, motion=motion, face_box=face_box)
    elapsed = time.perf_counter() - start
    apply_repeats = 5
    apply_start = time.perf_counter()
    for _ in range(apply_repeats):
        for depth, params in zip(video_depth, result.parameters):
            sync.apply_frame(depth, params)
    apply_ms_per_frame = (time.perf_counter() - apply_start) * 1000.0 / (apply_repeats * len(video_depth))
    fixed = np.maximum(result.scales[anchor] * video_depth + result.offsets[anchor], 0.0)
    face_mask = _region_mask(photo_low.shape, face_box)
    metrics: dict[str, float | int | str] = {
        "scene": scene,
        "frame_count": int(len(video_depth)),
        "anchor_index": anchor,
        "prepare_ms_total": elapsed * 1000.0,
        "prepare_ms_per_frame": elapsed * 1000.0 / len(video_depth),
        "apply_ms_per_frame": apply_ms_per_frame,
        "anchor_nmae": float(np.nanmedian(np.abs(result.depths[anchor] - photo_low)) / (_robust_range(photo_low) + 1e-6)),
        "raw_switch_jump_face": _aligned_switch_error(video_depth, photo_low, anchor, motion, face_box, True),
        "fixed_switch_jump_face": _aligned_switch_error(fixed, photo_low, anchor, motion, face_box, True),
        "synced_switch_jump_face": _aligned_switch_error(result.depths, photo_low, anchor, motion, face_box, True),
        "raw_switch_jump_global": _aligned_switch_error(video_depth, photo_low, anchor, motion, face_box, False),
        "fixed_switch_jump_global": _aligned_switch_error(fixed, photo_low, anchor, motion, face_box, False),
        "synced_switch_jump_global": _aligned_switch_error(result.depths, photo_low, anchor, motion, face_box, False),
        "raw_coc_jump": _aligned_switch_error(video_depth, photo_low, anchor, motion, face_box, False, True),
        "fixed_coc_jump": _aligned_switch_error(fixed, photo_low, anchor, motion, face_box, False, True),
        "synced_coc_jump": _aligned_switch_error(result.depths, photo_low, anchor, motion, face_box, False, True),
        "scale_jitter_median": float(np.median(np.abs(np.diff(result.scales)) / np.maximum(np.abs(result.scales[:-1]), 1e-6))),
        "offset_jitter_median": float(np.median(np.abs(np.diff(result.offsets))) / (_robust_range(photo_low) + 1e-6)),
        "fallback_frames": int(sum(bool(reason) for reason in result.fallback_reasons)),
    }
    np.savez_compressed(result_dir / "synced_depth.npz", disparity=result.depths)
    with (result_dir / "parameters.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "scale", "offset", "confidence", "fallback_reason"))
        for index, params in enumerate(result.parameters):
            writer.writerow((index, params.scale, params.offset, params.confidence, params.fallback_reason))
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _render_comparison(clip_dir / "clip.mp4", result_dir / "comparison.mp4", fixed, result.depths, photo_low, anchor, face_mask)
    return metrics


def write_report(metrics: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DepthSync 三视频验证结果",
        "",
        "| 场景 | 锚点 NMAE | 人脸跳变 raw → sync | fixed → sync | CoC 跳变 raw → sync | 参数估计 ms/帧 | 播放乘加 ms/帧 | 锁定/回退帧 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        lines.append(
            f"| {item['scene']} | {item['anchor_nmae']:.4f} | "
            f"{item['raw_switch_jump_face']:.4f} → {item['synced_switch_jump_face']:.4f} | "
            f"{item['fixed_switch_jump_face']:.4f} → {item['synced_switch_jump_face']:.4f} | "
            f"{item['raw_coc_jump']:.4f} → {item['synced_coc_jump']:.4f} | "
            f"{item['prepare_ms_per_frame']:.3f} | {item['apply_ms_per_frame']:.3f} | {item['fallback_frames']} |"
        )
    face_reductions = [
        100.0 * (1.0 - float(item["synced_switch_jump_face"]) / max(float(item["raw_switch_jump_face"]), 1e-9))
        for item in metrics
    ]
    coc_reductions = [
        100.0 * (1.0 - float(item["synced_coc_jump"]) / max(float(item["raw_coc_jump"]), 1e-9))
        for item in metrics
    ]
    fixed_non_regression = all(
        float(item["synced_switch_jump_face"]) <= float(item["fixed_switch_jump_face"]) + 1e-6 for item in metrics
    )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 三段视频的人脸尺度切换误差相对未对齐 VDA 分别下降 {', '.join(f'{value:.1f}%' for value in face_reductions)}。",
            f"- CoC 代理跳变分别下降 {', '.join(f'{value:.1f}%' for value in coc_reductions)}。",
            f"- 锚点锁定后的 fixed→sync 非退化检查：{'通过' if fixed_non_regression else '未通过'}。",
            f"- Python/NumPy 播放乘加最大耗时 {max(float(item['apply_ms_per_frame']) for item in metrics):.3f} ms/帧；拍后参数估计最大耗时 {max(float(item['prepare_ms_per_frame']) for item in metrics):.3f} ms/帧。",
            "",
            "## 口径与限制",
            "",
            "- 每段视频取时间中点附近 3 秒，按时间戳统一为 30 fps / 90 帧，照片锚点为第 45 帧。",
            "- DepthPro 只处理锚帧，Video Depth Anything Small 使用 streaming 模式处理全部视频帧。",
            "- 测试素材没有端侧 ISP MV，使用 18×32 稀疏 LK 网格模拟；人脸框为锚帧人工配置。",
            "- raw→sync 数值表示未标定 disparity 与照片 disparity 的切换不连续性，不代表绝对深度精度。",
            "- 本测试没有真实深度 GT，主要验收尺度一致性、CoC 连续性、非退化回退和运行开销。",
            "",
            "完整对比视频、逐帧参数和深度缓存位于本地 `results/<scene>/`，不进入 Git。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-root", type=Path, default=Path("artifacts/clips"))
    parser.add_argument("--depth-root", type=Path, default=Path("artifacts/depth"))
    parser.add_argument("--result-root", type=Path, default=Path("results"))
    parser.add_argument("--scenes", nargs="+", default=["01", "02", "03"])
    parser.add_argument("--scene-config", type=Path, default=Path("config/validation-scenes.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/validation.md"))
    args = parser.parse_args()
    scene_config = json.loads(args.scene_config.read_text(encoding="utf-8"))
    metrics = [
        evaluate_scene(scene, args.clip_root, args.depth_root, args.result_root, tuple(scene_config[scene]["face_box"]))
        for scene in args.scenes
    ]
    write_report(metrics, args.report)


if __name__ == "__main__":
    main()
