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


def _temporal_errors(sequence: np.ndarray, motion: MotionSequence, scale: float) -> np.ndarray:
    """Motion-compensated median frame differences, indexed by destination frame - 1."""
    height, width = sequence.shape[1:]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    errors: list[float] = []
    for index in range(1, len(sequence)):
        field = cv2.resize(motion.to_previous[index], (width, height), interpolation=cv2.INTER_LINEAR)
        previous = cv2.remap(
            sequence[index - 1],
            xx + field[..., 0] * max(width - 1, 1),
            yy + field[..., 1] * max(height - 1, 1),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
        confidence = cv2.resize(
            motion.confidence_previous[index], (width, height), interpolation=cv2.INTER_LINEAR
        )
        valid = np.isfinite(previous) & np.isfinite(sequence[index]) & (confidence > 0.15)
        errors.append(
            float(np.median(np.abs(sequence[index][valid] - previous[valid])) / scale)
            if np.any(valid)
            else float("nan")
        )
    return np.asarray(errors, np.float32)


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
    static_box: tuple[float, float, float, float] | None = None,
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
            static_grid_shape=(0, 0),
            static_target_shape=(0, 0),
            residual_radius=0,
        )
    )
    v3_sync = DepthSync(DepthSyncConfig(depth_mode="disparity"))
    v1, v1_prepare_ms, v1_apply_ms = _run(v1_sync, video_depth, photo, anchor, motion, face_box)
    v3, v3_prepare_ms, v3_apply_ms = _run(v3_sync, video_depth, photo, anchor, motion, face_box)
    temporal_scale = _robust_range(photo_low) + 1e-6
    v1_temporal = _temporal_errors(v1.depths, motion, temporal_scale)
    v3_temporal = _temporal_errors(v3.depths, motion, temporal_scale)
    interior = slice(4, max(len(v3_temporal) - 5, 5))
    excess = v3_temporal[interior] - v1_temporal[interior]
    excess_index = int(np.nanargmax(excess)) + 5

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
        "v1_temporal_p95": float(np.nanquantile(v1_temporal[interior], 0.95)),
        "v3_temporal_p95": float(np.nanquantile(v3_temporal[interior], 0.95)),
        "v3_excess_jump_max": float(np.nanmax(excess)),
        "v3_excess_jump_frame": excess_index,
        "v1_prepare_ms_per_frame": v1_prepare_ms,
        "v3_prepare_ms_per_frame": v3_prepare_ms,
        "v1_apply_ms_per_frame": v1_apply_ms,
        "v3_apply_ms_per_frame": v3_apply_ms,
        "v3_parameter_bytes": int(
            v3.lut_x.nbytes + v3.lut_y.nbytes + v3.residual_grids.nbytes
            + v3.static_mask.nbytes + v3.static_target_grid.nbytes
            + np.dtype(np.float32).itemsize
        ),
        "v3_fallback_frames": int(sum(bool(reason) for reason in v3.fallback_reasons)),
    }
    if static_box is not None:
        x0, y0, x1, y1 = static_box
        ys = slice(int(y0 * video_depth.shape[1]), int(y1 * video_depth.shape[1]))
        xs = slice(int(x0 * video_depth.shape[2]), int(x1 * video_depth.shape[2]))
        photo_median = float(np.nanmedian(photo_low[ys, xs]))
        v1_medians = np.nanmedian(v1.depths[:, ys, xs], axis=(1, 2))
        v3_medians = np.nanmedian(v3.depths[:, ys, xs], axis=(1, 2))
        metrics.update(
            {
                "static_photo_median": photo_median,
                "v1_static_median_range": float(np.nanmax(v1_medians) - np.nanmin(v1_medians)),
                "v3_static_median_range": float(np.nanmax(v3_medians) - np.nanmin(v3_medians)),
                "v3_static_photo_bias": float(np.nanmedian(np.abs(v3_medians - photo_median))),
            }
        )
        mask_full = cv2.resize(
            v3.static_mask,
            (video_depth.shape[2], video_depth.shape[1]),
            interpolation=cv2.INTER_LINEAR,
        )
        margin_y = max((ys.stop - ys.start) // 8, 1)
        margin_x = max((xs.stop - xs.start) // 8, 1)
        mask_inner = mask_full[
            ys.start + margin_y : ys.stop - margin_y,
            xs.start + margin_x : xs.stop - margin_x,
        ]
        metrics["v3_static_mask_laplacian_p95"] = float(
            np.nanquantile(np.abs(cv2.Laplacian(mask_inner, cv2.CV_32F)), 0.95)
        )
    np.savez_compressed(result_dir / "affine_depth.npz", disparity=v1.depths)
    np.savez_compressed(result_dir / "synced_depth.npz", disparity=v3.depths)
    np.savez_compressed(
        result_dir / "temporal_errors.npz",
        v1=v1_temporal,
        v3=v3_temporal,
    )
    np.savez_compressed(
        result_dir / "v3_parameters.npz",
        scales=v3.scales,
        offsets=v3.offsets,
        confidences=v3.confidences,
        lut_x=v3.lut_x,
        lut_y=v3.lut_y,
        residual_grids=v3.residual_grids,
        static_mask=v3.static_mask,
        static_target_grid=v3.static_target_grid,
        guidance_range=np.float32(v3.guidance_range),
        fallback_reasons=np.asarray(v3.fallback_reasons),
    )
    with (result_dir / "parameters.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "lut_scale_correction", "lut_offset_correction", "confidence", "fallback_reason"))
        for index, params in enumerate(v3.parameters):
            writer.writerow((index, params.scale, params.offset, params.confidence, params.fallback_reason))
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def write_report(metrics: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DepthSync V3.5 时域稳定性验证结果",
        "",
        "| 场景 | 锚帧 NMAE V1→V3 | 人脸切换 V1→V3 | 全局切换 V1→V3 | 时序 P95 V1→V3 | 最大新增跳变（帧） | V3 ms/帧 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        lines.append(
            f"| {item['scene']} | {item['v1_anchor_nmae']:.4f} → {item['v3_anchor_nmae']:.4f} | "
            f"{item['v1_switch_face']:.4f} → {item['v3_switch_face']:.4f} | "
            f"{item['v1_switch_global']:.4f} → {item['v3_switch_global']:.4f} | "
            f"{item['v1_temporal_p95']:.4f} → {item['v3_temporal_p95']:.4f} | "
            f"{item['v3_excess_jump_max']:.4f}（{item['v3_excess_jump_frame']}） | "
            f"{item['v3_apply_ms_per_frame']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 问题定位与修复",
            "",
            "- 旧版在每帧重新拟合完整 LUT，输入分布轻微变化会改变 LUT 横轴和节点；02 的第 23 帧因此出现映射突变。",
            "- 锚点锁定结束后直接恢复高置信度更新，03 在锚点后的第 47 帧出现单帧参数跳变。",
            "- 旧版残差只在 ±15 帧衰减，权重变化和累计 MV 噪声进一步放大了相邻帧差异。",
            "- V3.1 固定锚帧 LUT 的非线性形状，逐帧只允许受限 affine 修正；锚点附近渐进解锁，并把残差衰减扩展到 ±30 帧。",
            "- V3.2 将低运动、低相对深度变化区域识别为静态背景，播放时直接复用照片的 32×18 静态目标层，消除固定墙面的慢漂移；动态区域仍使用 V3.1 路径。",
            "- V3.3 增加时序范围门控、照片深度边缘腐蚀和逐帧深度边缘保护；静态层不再侵入人物轮廓，粗残差也不会跨前后景混合。",
            "- V3.4 对静态权重做小孔闭合与低通正则，再重新施加动态占用和深度边缘保护，消除 block MV 置信度孔洞在平面墙上形成的块状分层。",
            "- V3.5 将静态照片目标层提升到 128×72，并对 8-bit 对比视频使用固定亚 LSB 抖动；前者减少低分辨率目标层的分段插值，后者只消除可视化量化产生的伪轮廓，不改变浮点深度。",
            "",
            "## 01 左墙专项检查",
            "",
            f"- V1 墙面中位值全片范围：{metrics[0].get('v1_static_median_range', float('nan')):.4f}。",
            f"- V3.5 墙面中位值全片范围：{metrics[0].get('v3_static_median_range', float('nan')):.4f}；相对 DepthPro 中位值偏差：{metrics[0].get('v3_static_photo_bias', float('nan')):.4f}。",
            f"- 墙面静态权重 Laplacian P95：{metrics[0].get('v3_static_mask_laplacian_p95', float('nan')):.6f}（越低表示网格分层越弱）。",
            "",
            "## 口径",
            "",
            "- V1 是逐帧全局 affine；V3.5 固定锚帧 8 节点单调 LUT 的形状，只允许逐帧小幅 affine 修正，动态区域叠加 16×9 残差网格，静态区域使用 64×36 掩码和全片共享的 128×72 照片目标层。",
            "- LUT 修正在锚点附近渐进解锁；残差通过稀疏 block MV 传播，在 ±30 帧内使用余弦权重衰减。",
            "- 时序指标是 block MV 补偿后的相邻帧中位差，P95 和最大新增跳变忽略首尾各 5 帧的流式模型启动/结束区。",
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
        evaluate_scene(
            scene,
            args.clip_root,
            args.depth_root,
            args.result_root,
            tuple(scene_config[scene]["face_box"]),
            tuple(scene_config[scene]["static_box"]) if "static_box" in scene_config[scene] else None,
        )
        for scene in args.scenes
    ]
    write_report(metrics, args.report)


if __name__ == "__main__":
    main()
