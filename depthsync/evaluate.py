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
from .anchors import load_anchor_disparity
from .flow import load_dense_flow
from .local_alignment import (
    LocalAlignmentConfig,
    apply_local_sequence,
    fit_flow_guided_fields,
    save_local_fields,
    select_temporal_safe_fields,
)
from .motion import estimate_block_motion, load_motion, save_motion
from .registration import estimate_photo_registration, remap_photo_depth


def _region_mask(shape: tuple[int, int], box: tuple[float, float, float, float]) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[:height, :width]
    return (xx >= x0 * width) & (xx < x1 * width) & (yy >= y0 * height) & (yy < y1 * height)


def temporal_frame_indices(
    frames: list[int] | tuple[int, ...] | None,
    frame_count: int,
) -> np.ndarray:
    """Map destination frame numbers to indices in a T-1 temporal-error array."""
    if frames is None:
        return np.zeros(0, np.int32)
    return np.asarray(
        sorted({int(frame) - 1 for frame in frames if 1 <= int(frame) < frame_count}),
        np.int32,
    )


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


def _temporal_errors(
    sequence: np.ndarray,
    motion: MotionSequence,
    scale: float,
    region: np.ndarray | None = None,
) -> np.ndarray:
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
        if region is not None:
            valid &= region
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
         face_box: tuple[float, float, float, float], photo_to_video_grid: np.ndarray | None = None) -> tuple[SyncResult, float, float]:
    start = time.perf_counter()
    result = sync.offline_prepare(
        video_depth,
        photo,
        anchor,
        motion=motion,
        face_box=face_box,
        photo_to_video_grid=photo_to_video_grid,
    )
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
    subject_box: tuple[float, float, float, float] | None = None,
    tail_frames: list[int] | None = None,
    anchor_model: str = "dav2-large",
    flow_root: Path = Path("artifacts/flow"),
) -> dict[str, float | int | str]:
    clip_dir = clip_root / scene
    depth_dir = depth_root / scene
    result_dir = result_root / scene
    result_dir.mkdir(parents=True, exist_ok=True)
    video_depth = np.load(depth_dir / "video_disparity.npz")["disparity"].astype(np.float32)
    photo = load_anchor_disparity(depth_dir, anchor_model)
    anchor = int(json.loads((clip_dir / "manifest.json").read_text(encoding="utf-8"))["anchor_index"])
    capture = cv2.VideoCapture(str(clip_dir / "clip.mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, anchor)
    ok, video_anchor_rgb = capture.read()
    capture.release()
    photo_rgb = cv2.imread(str(clip_dir / "anchor.png"), cv2.IMREAD_COLOR)
    if not ok or photo_rgb is None:
        raise ValueError(f"Cannot read RGB anchor for scene {scene}")
    registration = estimate_photo_registration(
        photo_rgb, video_anchor_rgb, video_depth.shape[1:]
    )
    photo_low = remap_photo_depth(photo, registration, video_depth.shape[1:])
    motion_path = result_dir / "motion.npz"
    if motion_path.exists():
        motion = load_motion(motion_path)
    else:
        motion = estimate_block_motion(clip_dir / "clip.mp4")
        save_motion(motion_path, motion)

    v4_sync = DepthSync(DepthSyncConfig(depth_mode="disparity"))
    v4, v4_prepare_ms, v4_apply_ms = _run(
        v4_sync, video_depth, photo, anchor, motion, face_box, registration.grid
    )
    dense_flow_path = flow_root / scene / "sea_raft_s_flow.npz"
    if not dense_flow_path.is_file():
        raise FileNotFoundError(
            f"Flow-guided alignment requires cached flow: {dense_flow_path}"
        )
    dense_flow = load_dense_flow(dense_flow_path)
    local_config = LocalAlignmentConfig()
    start = time.perf_counter()
    local_fields = fit_flow_guided_fields(
        v4.depths, photo_low, anchor, dense_flow, local_config
    )
    local_fields, local_strength, local_safety = select_temporal_safe_fields(
        v4.depths,
        local_fields,
        dense_flow,
        local_config,
        photo_anchor=photo_low,
        anchor_index=anchor,
        priority_box=face_box,
    )
    local_prepare_ms = (time.perf_counter() - start) * 1000.0 / len(video_depth)
    repeats = 5
    start = time.perf_counter()
    for _ in range(repeats):
        local_depths = apply_local_sequence(v4.depths, local_fields, local_config)
    local_apply_ms = (time.perf_counter() - start) * 1000.0 / (repeats * len(video_depth))
    temporal_scale = _robust_range(photo_low) + 1e-6
    raw_temporal = _temporal_errors(video_depth, motion, temporal_scale)
    v4_temporal = _temporal_errors(v4.depths, motion, temporal_scale)
    local_temporal = _temporal_errors(local_depths, motion, temporal_scale)
    if subject_box is None:
        x0, y0, x1, y1 = face_box
        subject_box = (
            max(x0 - 0.20, 0.0),
            max(y0 - 0.20, 0.0),
            min(x1 + 0.20, 1.0),
            min(y1 + 0.50, 1.0),
        )
    subject_region = _region_mask(photo_low.shape, subject_box)
    raw_subject_temporal = _temporal_errors(
        video_depth, motion, temporal_scale, subject_region
    )
    v4_subject_temporal = _temporal_errors(
        v4.depths, motion, temporal_scale, subject_region
    )
    local_subject_temporal = _temporal_errors(
        local_depths, motion, temporal_scale, subject_region
    )
    interior = slice(4, max(len(v4_temporal) - 5, 5))
    tail_indices = temporal_frame_indices(tail_frames, len(video_depth))
    subject_temporal_selection: slice | np.ndarray = (
        tail_indices if tail_indices.size else interior
    )
    excess = v4_temporal[interior] - raw_temporal[interior]
    excess_index = int(np.nanargmax(excess)) + 5
    local_excess = local_temporal[interior] - v4_temporal[interior]
    local_excess_index = int(np.nanargmax(local_excess)) + 5

    metrics: dict[str, float | int | str] = {
        "scene": scene,
        "anchor_model": anchor_model,
        "frame_count": int(len(video_depth)),
        "anchor_index": anchor,
        "raw_anchor_nmae": _anchor_nmae(video_depth[anchor], photo_low),
        "v4_anchor_nmae": _anchor_nmae(v4.depths[anchor], photo_low),
        "local_anchor_nmae": _anchor_nmae(local_depths[anchor], photo_low),
        "raw_anchor_edge_nmae": _anchor_edge_nmae(video_depth[anchor], photo_low),
        "v4_anchor_edge_nmae": _anchor_edge_nmae(v4.depths[anchor], photo_low),
        "local_anchor_edge_nmae": _anchor_edge_nmae(local_depths[anchor], photo_low),
        "raw_switch_face": _aligned_switch_error(video_depth, photo_low, anchor, motion, face_box, True),
        "v4_switch_face": _aligned_switch_error(v4.depths, photo_low, anchor, motion, face_box, True),
        "local_switch_face": _aligned_switch_error(local_depths, photo_low, anchor, motion, face_box, True),
        "raw_switch_global": _aligned_switch_error(video_depth, photo_low, anchor, motion, face_box, False),
        "v4_switch_global": _aligned_switch_error(v4.depths, photo_low, anchor, motion, face_box, False),
        "local_switch_global": _aligned_switch_error(local_depths, photo_low, anchor, motion, face_box, False),
        "raw_temporal_p95": float(np.nanquantile(raw_temporal[interior], 0.95)),
        "v4_temporal_p95": float(np.nanquantile(v4_temporal[interior], 0.95)),
        "local_temporal_p95": float(np.nanquantile(local_temporal[interior], 0.95)),
        "raw_subject_temporal_p95": float(
            np.nanquantile(raw_subject_temporal[subject_temporal_selection], 0.95)
        ),
        "v4_subject_temporal_p95": float(
            np.nanquantile(v4_subject_temporal[subject_temporal_selection], 0.95)
        ),
        "local_subject_temporal_p95": float(
            np.nanquantile(local_subject_temporal[subject_temporal_selection], 0.95)
        ),
        "v4_excess_jump_max": float(np.nanmax(excess)),
        "v4_excess_jump_frame": excess_index,
        "local_excess_jump_max": float(np.nanmax(local_excess)),
        "local_excess_jump_frame": local_excess_index,
        "v4_prepare_ms_per_frame": v4_prepare_ms,
        "v4_apply_ms_per_frame": v4_apply_ms,
        "local_prepare_ms_per_frame": local_prepare_ms,
        "local_apply_ms_per_frame": local_apply_ms,
        "v4_parameter_bytes": int(
            v4.lut_x.nbytes + v4.lut_y.nbytes
            + v4.scales.nbytes + v4.offsets.nbytes + v4.confidences.nbytes
        ),
        "v4_fallback_frames": int(sum(bool(reason) for reason in v4.fallback_reasons)),
        "local_parameter_bytes": int(
            local_fields.delta_scale.astype(np.float16).nbytes
            + local_fields.offset_norm.astype(np.float16).nbytes
            + local_fields.confidence.astype(np.float16).nbytes
            + local_fields.guide_depth.astype(np.float16).nbytes
        ),
        "local_supported_node_fraction": float(np.mean(local_fields.confidence > 0.0)),
        "local_strength": local_strength,
        **local_safety,
        "registration_method": registration.method,
        "registration_confidence": registration.confidence,
        "registration_identity_error": registration.identity_error,
        "registration_error": registration.registered_error,
    }
    if static_box is not None:
        x0, y0, x1, y1 = static_box
        ys = slice(int(y0 * video_depth.shape[1]), int(y1 * video_depth.shape[1]))
        xs = slice(int(x0 * video_depth.shape[2]), int(x1 * video_depth.shape[2]))
        photo_median = float(np.nanmedian(photo_low[ys, xs]))
        raw_medians = np.nanmedian(video_depth[:, ys, xs], axis=(1, 2))
        v4_medians = np.nanmedian(v4.depths[:, ys, xs], axis=(1, 2))
        local_medians = np.nanmedian(local_depths[:, ys, xs], axis=(1, 2))
        metrics.update(
            {
                "static_photo_median": photo_median,
                "raw_static_median_range": float(np.nanmax(raw_medians) - np.nanmin(raw_medians)),
                "v4_static_median_range": float(np.nanmax(v4_medians) - np.nanmin(v4_medians)),
                "v4_static_photo_bias": float(np.nanmedian(np.abs(v4_medians - photo_median))),
                "local_static_median_range": float(np.nanmax(local_medians) - np.nanmin(local_medians)),
                "local_static_photo_bias": float(np.nanmedian(np.abs(local_medians - photo_median))),
            }
        )
        edge_errors: list[float] = []
        fallback_widths: list[int] = []
        edge_threshold = 0.01 * temporal_scale
        for y in range(ys.start, ys.stop):
            row_gradient = np.abs(np.diff(photo_low[y, xs]))
            if row_gradient.size == 0:
                continue
            boundary = xs.start + int(np.argmax(row_gradient))
            error = np.abs(v4.depths[anchor, y, xs.start:boundary] - photo_low[y, xs.start:boundary])
            if error.size == 0:
                continue
            edge_errors.extend(error.tolist())
            width = 0
            for is_fallback in (error > edge_threshold)[::-1]:
                if is_fallback:
                    width += 1
                elif width:
                    break
            fallback_widths.append(width)
        metrics["v4_static_edge_mae"] = float(np.mean(edge_errors)) if edge_errors else float("nan")
        metrics["v4_static_edge_p95"] = (
            float(np.quantile(edge_errors, 0.95)) if edge_errors else float("nan")
        )
        metrics["v4_static_edge_fallback_width_median"] = (
            float(np.median(fallback_widths)) if fallback_widths else float("nan")
        )
    np.savez_compressed(result_dir / "v4_depth.npz", disparity=v4.depths)
    np.savez_compressed(result_dir / "v41_local_depth.npz", disparity=local_depths)
    save_local_fields(str(result_dir / "v41_local_parameters.npz"), local_fields)
    np.savez_compressed(
        result_dir / "temporal_errors.npz",
        raw=raw_temporal,
        v4=v4_temporal,
        local=local_temporal,
    )
    np.savez_compressed(
        result_dir / "v4_parameters.npz",
        scales=v4.scales,
        offsets=v4.offsets,
        confidences=v4.confidences,
        lut_x=v4.lut_x,
        lut_y=v4.lut_y,
        fallback_reasons=np.asarray(v4.fallback_reasons),
    )
    with (result_dir / "parameters.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "lut_scale_correction", "lut_offset_correction", "confidence", "fallback_reason"))
        for index, params in enumerate(v4.parameters):
            writer.writerow((index, params.scale, params.offset, params.confidence, params.fallback_reason))
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def write_report(metrics: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DepthSync V4.1 全局 LUT＋光流局部参数场验证结果",
        "",
        "| 场景 | 锚帧 NMAE raw→LUT→Local | 人脸切换 raw→LUT→Local | 全局切换 raw→LUT→Local | 时序 P95 raw→LUT→Local | 主体时序 raw→LUT→Local | Local新增跳变（帧） | Local准备/应用 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        lines.append(
            f"| {item['scene']} | {item['raw_anchor_nmae']:.4f} → {item['v4_anchor_nmae']:.4f} → {item['local_anchor_nmae']:.4f} | "
            f"{item['raw_switch_face']:.4f} → {item['v4_switch_face']:.4f} → {item['local_switch_face']:.4f} | "
            f"{item['raw_switch_global']:.4f} → {item['v4_switch_global']:.4f} → {item['local_switch_global']:.4f} | "
            f"{item['raw_temporal_p95']:.4f} → {item['v4_temporal_p95']:.4f} → {item['local_temporal_p95']:.4f} | "
            f"{item['raw_subject_temporal_p95']:.4f} → {item['v4_subject_temporal_p95']:.4f} → {item['local_subject_temporal_p95']:.4f} | "
            f"{item['local_excess_jump_max']:.4f}（{item['local_excess_jump_frame']}） | "
            f"{item['local_prepare_ms_per_frame']:.2f}/{item['local_apply_ms_per_frame']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 处理口径",
            "",
            "- V4.1 在锚帧自适应选择 8/16/32/64 节点的全局单调 LUT，并统一为 64 节点定长参数。",
            "- 局部增强使用 SEA-RAFT-S 双向光流建立当前帧到照片锚帧的可信对应，只拟合 8×12 scale/shift 参数，不直接传播照片残差图。",
            "- 遮挡、越界、场景切换和低置信度位置严格回退到 V4.1 全局 LUT。",
            "- 播放阶段执行一维 LUT 和低分辨率参数场采样，不运行光流、不读取照片深度。",
            "- 低相关或分布突变帧冻结到稳定映射，避免大遮挡污染全局值域。",
            "- 时序指标是 block MV 补偿后的相邻帧中位差，P95 和最大新增跳变忽略首尾各 5 帧的流式模型启动/结束区。",
            "- 主体时序指标在扩展人脸 ROI 内计算，用于单独约束全局映射对人物一致性的影响。",
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
    parser.add_argument("--flow-root", type=Path, default=Path("artifacts/flow"))
    parser.add_argument("--anchor-model", choices=("dav2-large", "depthpro"), default="dav2-large")
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
            tuple(scene_config[scene]["subject_box"]) if "subject_box" in scene_config[scene] else None,
            list(scene_config[scene]["tail_frames"]) if "tail_frames" in scene_config[scene] else None,
            anchor_model=args.anchor_model,
            flow_root=args.flow_root,
        )
        for scene in args.scenes
    ]
    write_report(metrics, args.report)


if __name__ == "__main__":
    main()
