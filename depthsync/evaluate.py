"""Offline metrics for the lightweight DepthSync validation clips."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .core import DepthSync, DepthSyncConfig, MotionSequence, SyncResult
from .motion import estimate_block_motion, load_motion, save_motion


def _region_mask(shape: tuple[int, int], box: tuple[float, float, float, float]) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[:height, :width]
    return (xx >= x0 * width) & (xx < x1 * width) & (yy >= y0 * height) & (yy < y1 * height)


def _robust_range(depth: np.ndarray) -> float:
    valid = depth[np.isfinite(depth)]
    return float(np.quantile(valid, 0.9) - np.quantile(valid, 0.1)) if valid.size else 1.0


def _anchor_nmae(depth: np.ndarray, photo: np.ndarray) -> float:
    valid = np.isfinite(depth) & np.isfinite(photo)
    return float(np.nanmedian(np.abs(depth[valid] - photo[valid])) / (_robust_range(photo) + 1e-6))


def _anchor_edge_nmae(depth: np.ndarray, photo: np.ndarray) -> float:
    def gradient(image: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    depth_gradient, photo_gradient = gradient(depth), gradient(photo)
    valid = np.isfinite(depth_gradient) & np.isfinite(photo_gradient)
    photo_scale = float(np.quantile(photo_gradient[valid], 0.9)) + 1e-6
    return float(np.median(np.abs(depth_gradient[valid] - photo_gradient[valid])) / photo_scale)


def _aligned_switch_error(
    sequence: np.ndarray,
    photo: np.ndarray,
    anchor: int,
    motion: MotionSequence,
    face_box: tuple[float, float, float, float],
    face_only: bool,
) -> float:
    """Compare both anchor neighbors against motion-warped photo depth."""
    height, width = photo.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    scale = _robust_range(photo) + 1e-6
    errors: list[float] = []
    for index, field, confidence in (
        (anchor - 1, motion.to_next[anchor - 1], motion.confidence_next[anchor - 1]),
        (anchor + 1, motion.to_previous[anchor + 1], motion.confidence_previous[anchor + 1]),
    ):
        dense = cv2.resize(field, (width, height), interpolation=cv2.INTER_LINEAR)
        map_x = xx + dense[..., 0] * max(width - 1, 1)
        map_y = yy + dense[..., 1] * max(height - 1, 1)
        reference = cv2.remap(
            photo, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan
        )
        conf = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
        valid = np.isfinite(reference) & np.isfinite(sequence[index]) & (conf > 0.15)
        if face_only:
            x0, y0, x1, y1 = face_box
            valid &= (
                (map_x >= x0 * width)
                & (map_x < x1 * width)
                & (map_y >= y0 * height)
                & (map_y < y1 * height)
            )
        difference = np.abs(sequence[index] - reference) / scale
        errors.append(float(np.nanmedian(difference[valid])) if np.any(valid) else float("nan"))
    return float(np.nanmean(errors))


def _run(sync: DepthSync, video_depth: np.ndarray, photo: np.ndarray, anchor: int, motion: MotionSequence,
         face_box: tuple[float, float, float, float]) -> tuple[SyncResult, float, float]:
    start = time.perf_counter()
    result = sync.offline_prepare(video_depth, photo, anchor, motion=motion, face_box=face_box)
    prepare_ms = (time.perf_counter() - start) * 1000.0 / len(video_depth)
    repeats = 5
    start = time.perf_counter()
    for _ in range(repeats):
        for depth, params in zip(video_depth, result.parameters):
            sync.apply_frame(depth, params)
    apply_ms = (time.perf_counter() - start) * 1000.0 / (repeats * len(video_depth))
    return result, prepare_ms, apply_ms


def evaluate_scene(
    scene: str,
    clip_root: Path,
    depth_root: Path,
    result_root: Path,
    face_box: tuple[float, float, float, float],
) -> dict[str, float | int | str]:
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

    v1_sync = DepthSync(
        DepthSyncConfig(
            depth_mode="disparity",
            mapping_mode="affine",
            residual_grid_shape=(0, 0),
            residual_radius=0,
        )
    )
    v3_sync = DepthSync(DepthSyncConfig(depth_mode="disparity"))
    v1, v1_prepare_ms, v1_apply_ms = _run(v1_sync, video_depth, photo, anchor, motion, face_box)
    v3, v3_prepare_ms, v3_apply_ms = _run(v3_sync, video_depth, photo, anchor, motion, face_box)

    metrics: dict[str, float | int | str] = {
        "scene": scene,
        "frame_count": int(len(video_depth)),
        "anchor_index": anchor,
        "raw_anchor_nmae": _anchor_nmae(video_depth[anchor], photo_low),
        "v1_anchor_nmae": _anchor_nmae(v1.depths[anchor], photo_low),
        "v3_anchor_nmae": _anchor_nmae(v3.depths[anchor], photo_low),
        "v1_anchor_edge_nmae": _anchor_edge_nmae(v1.depths[anchor], photo_low),
        "v3_anchor_edge_nmae": _anchor_edge_nmae(v3.depths[anchor], photo_low),
        "raw_switch_face": _aligned_switch_error(video_depth, photo_low, anchor, motion, face_box, True),
        "v1_switch_face": _aligned_switch_error(v1.depths, photo_low, anchor, motion, face_box, True),
        "v3_switch_face": _aligned_switch_error(v3.depths, photo_low, anchor, motion, face_box, True),
        "raw_switch_global": _aligned_switch_error(video_depth, photo_low, anchor, motion, face_box, False),
        "v1_switch_global": _aligned_switch_error(v1.depths, photo_low, anchor, motion, face_box, False),
        "v3_switch_global": _aligned_switch_error(v3.depths, photo_low, anchor, motion, face_box, False),
        "v1_prepare_ms_per_frame": v1_prepare_ms,
        "v3_prepare_ms_per_frame": v3_prepare_ms,
        "v1_apply_ms_per_frame": v1_apply_ms,
        "v3_apply_ms_per_frame": v3_apply_ms,
        "v3_parameter_bytes": int(v3.lut_x.nbytes + v3.lut_y.nbytes + v3.residual_grids.nbytes),
        "v3_fallback_frames": int(sum(bool(reason) for reason in v3.fallback_reasons)),
    }
    np.savez_compressed(result_dir / "affine_depth.npz", disparity=v1.depths)
    np.savez_compressed(result_dir / "synced_depth.npz", disparity=v3.depths)
    np.savez_compressed(
        result_dir / "v3_parameters.npz",
        scales=v3.scales,
        offsets=v3.offsets,
        confidences=v3.confidences,
        lut_x=v3.lut_x,
        lut_y=v3.lut_y,
        residual_grids=v3.residual_grids,
        fallback_reasons=np.asarray(v3.fallback_reasons),
    )
    with (result_dir / "parameters.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "scale", "offset", "confidence", "fallback_reason"))
        for index, params in enumerate(v3.parameters):
            writer.writerow((index, params.scale, params.offset, params.confidence, params.fallback_reason))
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def write_report(metrics: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DepthSync V3 三视频验证结果",
        "",
        "| 场景 | 锚帧 NMAE V1→V3 | 锚帧边缘 NMAE V1→V3 | 人脸切换误差 V1→V3 | 全局切换误差 V1→V3 | V3 应用 ms/帧 | 参数 KB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        lines.append(
            f"| {item['scene']} | {item['v1_anchor_nmae']:.4f} → {item['v3_anchor_nmae']:.4f} | "
            f"{item['v1_anchor_edge_nmae']:.4f} → {item['v3_anchor_edge_nmae']:.4f} | "
            f"{item['v1_switch_face']:.4f} → {item['v3_switch_face']:.4f} | "
            f"{item['v1_switch_global']:.4f} → {item['v3_switch_global']:.4f} | "
            f"{item['v3_apply_ms_per_frame']:.3f} | {float(item['v3_parameter_bytes']) / 1024.0:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 口径",
            "",
            "- V1 是逐帧全局 affine scale/offset；V3 是 8 节点单调 LUT 加 16×9 残差网格。",
            "- 残差从照片锚帧通过稀疏 block MV 向前后传播，在 ±15 帧内使用余弦权重衰减。",
            "- 每段素材取中间 3 秒并统一为 30 fps / 90 帧，照片锚点为第 45 帧。",
            "- 耗时为 Python/NumPy 原型实测，只用于相对比较；端侧 C/C++/NEON 实现会采用定长 LUT。",
            "- 本验证没有真实深度 GT，指标衡量照片锚点一致性和切换连续性，不代表绝对深度精度。",
            "- 深度对比视频、逐帧参数和深度缓存位于本地 `results/<scene>/`，不进入 Git。",
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
