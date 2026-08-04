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
from .flow import DenseFlowSequence, load_flow
from .motion import estimate_block_motion, load_motion, save_motion


def flat_correction_gradient_p99(
    synced: np.ndarray,
    base: np.ndarray,
    flat_threshold: float,
    region: np.ndarray | None = None,
) -> float:
    """P99 correction step on edges that are flat in the V4 base depth."""
    synced = np.asarray(synced, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    if synced.shape != base.shape or synced.ndim != 2:
        raise ValueError("synced and base must have equal [H,W] shapes")
    correction = synced - base
    values: list[np.ndarray] = []
    for axis in (0, 1):
        base_delta = np.diff(base, axis=axis)
        correction_delta = np.diff(correction, axis=axis)
        valid = (
            np.isfinite(base_delta)
            & np.isfinite(correction_delta)
            & (np.abs(base_delta) < flat_threshold)
        )
        if region is not None:
            region_pair = (
                region[:-1] & region[1:]
                if axis == 0
                else region[:, :-1] & region[:, 1:]
            )
            valid &= region_pair
        values.append(np.abs(correction_delta[valid]))
    finite = np.concatenate([value for value in values if value.size]) if any(
        value.size for value in values
    ) else np.zeros(0, np.float32)
    return float(np.quantile(finite, 0.99)) if finite.size else 0.0


def _gate(actual: float | int, limit: float | int) -> dict[str, float | int | bool]:
    return {"actual": actual, "limit": limit, "passed": bool(actual <= limit)}


def classify_v5_gates(
    scene: str,
    metrics: dict[str, float | int],
) -> dict[str, dict[str, float | int | bool]]:
    """Classify the numeric V5 acceptance gates without hiding scene failures."""
    gates: dict[str, dict[str, float | int | bool]] = {
        "all_parameter_bytes": _gate(
            metrics["v5_parameter_bytes"],
            1_244_160,
        )
    }
    if scene == "01":
        gates["01_subject_anchor_nmae"] = _gate(
            metrics["v5_subject_anchor_nmae"],
            0.80 * metrics["v4_subject_anchor_nmae"],
        )
        gates["01_wall_flat_correction_gradient_p99"] = _gate(
            metrics["v5_wall_flat_correction_gradient_p99"],
            1.10 * metrics["v4_wall_flat_correction_gradient_p99"] + 1e-6,
        )
    elif scene == "02":
        gates["02_subject_temporal"] = _gate(
            metrics["v5_subject_temporal_p95"],
            1.05 * metrics["v4_subject_temporal_p95"],
        )
        gates["02_revealed_background_confidence"] = _gate(
            metrics["v5_revealed_background_confidence_p95"],
            0.05,
        )
    elif scene == "03":
        gates["03_excess_jump_max"] = _gate(
            metrics["v5_excess_jump_max"],
            metrics["v4_excess_jump_max"] + 2e-4,
        )
    elif scene.isdigit() and 4 <= int(scene) <= 20:
        gates["generic_anchor_nmae"] = _gate(
            metrics["v5_anchor_nmae"],
            metrics["v4_anchor_nmae"],
        )
        gates["generic_subject_anchor_nmae"] = _gate(
            metrics["v5_subject_anchor_nmae"],
            metrics["v4_subject_anchor_nmae"],
        )
        gates["generic_temporal_p95"] = _gate(
            metrics["v5_temporal_p95"],
            1.10 * metrics["v4_temporal_p95"] + 2e-4,
        )
    else:
        raise ValueError(f"unknown validation scene: {scene}")
    return gates


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


def _read_rgb_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"Video has no readable frames: {path}")
    return np.stack(frames)


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
         face_box: tuple[float, float, float, float], rgb_frames: np.ndarray | None = None,
         dense_flow: DenseFlowSequence | None = None) -> tuple[SyncResult, float, float]:
    start = time.perf_counter()
    result = sync.offline_prepare(
        video_depth,
        photo,
        anchor,
        motion=motion,
        face_box=face_box,
        rgb_frames=rgb_frames,
        dense_flow=dense_flow,
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
    flow_root: Path | None = None,
    algorithm_version: str = "v4",
    anchor_model: str = "dav2-large",
) -> dict[str, float | int | str]:
    clip_dir = clip_root / scene
    depth_dir = depth_root / scene
    result_dir = result_root / scene
    result_dir.mkdir(parents=True, exist_ok=True)
    video_depth = np.load(depth_dir / "video_disparity.npz")["disparity"].astype(np.float32)
    photo = load_anchor_disparity(depth_dir, anchor_model)
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
            algorithm_version="v1",
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
    v5: SyncResult | None = None
    v5_prepare_ms = 0.0
    v5_apply_ms = 0.0
    dense_flow: DenseFlowSequence | None = None
    if algorithm_version == "v5":
        if flow_root is None:
            raise ValueError("flow_root is required for V5 evaluation")
        rgb_frames = _read_rgb_video(clip_dir / "clip.mp4")
        dense_flow = load_flow(flow_root / scene / "sea_raft_s_flow.npz")
        v5_sync = DepthSync(DepthSyncConfig(algorithm_version="v5", depth_mode="disparity"))
        v5, v5_prepare_ms, v5_apply_ms = _run(
            v5_sync,
            video_depth,
            photo,
            anchor,
            motion,
            face_box,
            rgb_frames,
            dense_flow,
        )
    temporal_scale = _robust_range(photo_low) + 1e-6
    v1_temporal = _temporal_errors(v1.depths, motion, temporal_scale)
    v3_temporal = _temporal_errors(v3.depths, motion, temporal_scale)
    if subject_box is None:
        x0, y0, x1, y1 = face_box
        subject_box = (
            max(x0 - 0.20, 0.0),
            max(y0 - 0.20, 0.0),
            min(x1 + 0.20, 1.0),
            min(y1 + 0.50, 1.0),
        )
    subject_region = _region_mask(photo_low.shape, subject_box)
    v1_subject_temporal = _temporal_errors(
        v1.depths, motion, temporal_scale, subject_region
    )
    v3_subject_temporal = _temporal_errors(
        v3.depths, motion, temporal_scale, subject_region
    )
    v5_temporal = None
    v5_subject_temporal = None
    if v5 is not None:
        v5_temporal = _temporal_errors(v5.depths, motion, temporal_scale)
        v5_subject_temporal = _temporal_errors(
            v5.depths,
            motion,
            temporal_scale,
            subject_region,
        )
    interior = slice(4, max(len(v3_temporal) - 5, 5))
    tail_indices = temporal_frame_indices(tail_frames, len(video_depth))
    subject_temporal_selection: slice | np.ndarray = (
        tail_indices if tail_indices.size else interior
    )
    excess = v3_temporal[interior] - v1_temporal[interior]
    excess_index = int(np.nanargmax(excess)) + 5

    metrics: dict[str, float | int | str] = {
        "scene": scene,
        "anchor_model": anchor_model,
        "frame_count": int(len(video_depth)),
        "anchor_index": anchor,
        "raw_anchor_nmae": _anchor_nmae(video_depth[anchor], photo_low),
        "v1_anchor_nmae": _anchor_nmae(v1.depths[anchor], photo_low),
        "v4_anchor_nmae": _anchor_nmae(v3.depths[anchor], photo_low),
        "v1_anchor_edge_nmae": _anchor_edge_nmae(v1.depths[anchor], photo_low),
        "v4_anchor_edge_nmae": _anchor_edge_nmae(v3.depths[anchor], photo_low),
        "raw_switch_face": _aligned_switch_error(video_depth, photo_low, anchor, motion, face_box, True),
        "v1_switch_face": _aligned_switch_error(v1.depths, photo_low, anchor, motion, face_box, True),
        "v4_switch_face": _aligned_switch_error(v3.depths, photo_low, anchor, motion, face_box, True),
        "raw_switch_global": _aligned_switch_error(video_depth, photo_low, anchor, motion, face_box, False),
        "v1_switch_global": _aligned_switch_error(v1.depths, photo_low, anchor, motion, face_box, False),
        "v4_switch_global": _aligned_switch_error(v3.depths, photo_low, anchor, motion, face_box, False),
        "v1_temporal_p95": float(np.nanquantile(v1_temporal[interior], 0.95)),
        "v4_temporal_p95": float(np.nanquantile(v3_temporal[interior], 0.95)),
        "v1_subject_temporal_p95": float(
            np.nanquantile(v1_subject_temporal[subject_temporal_selection], 0.95)
        ),
        "v4_subject_temporal_p95": float(
            np.nanquantile(v3_subject_temporal[subject_temporal_selection], 0.95)
        ),
        "v4_excess_jump_max": float(np.nanmax(excess)),
        "v4_excess_jump_frame": excess_index,
        "v1_prepare_ms_per_frame": v1_prepare_ms,
        "v4_prepare_ms_per_frame": v3_prepare_ms,
        "v1_apply_ms_per_frame": v1_apply_ms,
        "v4_apply_ms_per_frame": v3_apply_ms,
        "v4_parameter_bytes": int(
            v3.lut_x.nbytes + v3.lut_y.nbytes + v3.residual_grids.nbytes
            + v3.static_mask.nbytes + v3.static_target_grid.nbytes
            + v3.region_labels.nbytes + v3.region_scales.nbytes
            + v3.region_offsets.nbytes + v3.region_shifts.nbytes
            + np.dtype(np.float32).itemsize
        ),
        "v4_fallback_frames": int(sum(bool(reason) for reason in v3.fallback_reasons)),
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
                "v4_static_median_range": float(np.nanmax(v3_medians) - np.nanmin(v3_medians)),
                "v4_static_photo_bias": float(np.nanmedian(np.abs(v3_medians - photo_median))),
            }
        )
        guidance_mask = (
            v3.static_mask
            if v3.static_mask.size
            else (v3.region_labels > 0).astype(np.float32)
        )
        if guidance_mask.size:
            mask_full = cv2.resize(
                guidance_mask,
                (video_depth.shape[2], video_depth.shape[1]),
                interpolation=cv2.INTER_NEAREST,
            )
            margin_y = max((ys.stop - ys.start) // 8, 1)
            margin_x = max((xs.stop - xs.start) // 8, 1)
            mask_inner = mask_full[
                ys.start + margin_y : ys.stop - margin_y,
                xs.start + margin_x : xs.stop - margin_x,
            ]
            metrics["v4_static_mask_laplacian_p95"] = float(
                np.nanquantile(np.abs(cv2.Laplacian(mask_inner, cv2.CV_32F)), 0.95)
            )
        else:
            metrics["v4_static_mask_laplacian_p95"] = 0.0
        edge_errors: list[float] = []
        fallback_widths: list[int] = []
        edge_threshold = 0.01 * temporal_scale
        for y in range(ys.start, ys.stop):
            row_gradient = np.abs(np.diff(photo_low[y, xs]))
            if row_gradient.size == 0:
                continue
            boundary = xs.start + int(np.argmax(row_gradient))
            error = np.abs(v3.depths[anchor, y, xs.start:boundary] - photo_low[y, xs.start:boundary])
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
    if v5 is not None and v5.local_fields is not None:
        assert v5_temporal is not None and v5_subject_temporal is not None
        v5_excess = v5_temporal[interior] - v1_temporal[interior]
        v5_excess_index = int(np.nanargmax(v5_excess)) + 5
        subject_photo = photo_low[subject_region]
        subject_v4 = v3.depths[anchor][subject_region]
        subject_v5 = v5.depths[anchor][subject_region]
        flat_threshold = 0.01 * temporal_scale
        wall_region = (
            _region_mask(photo_low.shape, static_box)
            if static_box is not None
            else np.ones(photo_low.shape, bool)
        )
        revealed_confidence: list[np.ndarray] = []
        if dense_flow is not None:
            grid_h, grid_w = v5.local_fields.confidence.shape[1:]
            for frame_index in range(max(anchor + 1, 75), len(video_depth)):
                pair_confidence = cv2.resize(
                    dense_flow.confidence_previous[frame_index - 1].astype(np.float32),
                    (grid_w, grid_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                revealed = pair_confidence < 0.05
                if np.any(revealed):
                    revealed_confidence.append(
                        v5.local_fields.confidence[frame_index][revealed]
                    )
        revealed_values = (
            np.concatenate(revealed_confidence)
            if revealed_confidence
            else np.zeros(0, np.float32)
        )
        metrics.update(
            {
                "v4_subject_anchor_nmae": _anchor_nmae(subject_v4, subject_photo),
                "v5_subject_anchor_nmae": _anchor_nmae(subject_v5, subject_photo),
                "v5_anchor_nmae": _anchor_nmae(v5.depths[anchor], photo_low),
                "v5_temporal_p95": float(
                    np.nanquantile(v5_temporal[interior], 0.95)
                ),
                "v5_subject_temporal_p95": float(
                    np.nanquantile(v5_subject_temporal[subject_temporal_selection], 0.95)
                ),
                "v5_excess_jump_max": float(np.nanmax(v5_excess)),
                "v5_excess_jump_frame": v5_excess_index,
                "v4_wall_flat_correction_gradient_p99": flat_correction_gradient_p99(
                    v3.depths[anchor],
                    video_depth[anchor],
                    flat_threshold,
                    wall_region,
                ),
                "v5_wall_flat_correction_gradient_p99": flat_correction_gradient_p99(
                    v5.depths[anchor],
                    video_depth[anchor],
                    flat_threshold,
                    wall_region,
                ),
                "v5_revealed_background_confidence_p95": float(
                    np.quantile(revealed_values, 0.95)
                )
                if revealed_values.size
                else 0.0,
                "v5_prepare_ms_per_frame": v5_prepare_ms,
                "v5_apply_ms_per_frame": v5_apply_ms,
                "v5_parameter_bytes": int(
                    v5.local_fields.delta_scale.astype(np.float16).nbytes
                    + v5.local_fields.offset_norm.astype(np.float16).nbytes
                    + v5.local_fields.confidence.astype(np.float16).nbytes
                ),
            }
        )
        metrics["v5_gates"] = classify_v5_gates(scene, metrics)  # type: ignore[assignment]
    np.savez_compressed(result_dir / "affine_depth.npz", disparity=v1.depths)
    np.savez_compressed(result_dir / "synced_depth.npz", disparity=v3.depths)
    np.savez_compressed(result_dir / "v4_depth.npz", disparity=v3.depths)
    if v5 is not None and v5.local_fields is not None:
        np.savez_compressed(result_dir / "v5_depth.npz", disparity=v5.depths)
        np.savez_compressed(
            result_dir / "v5_parameters.npz",
            delta_scale=v5.local_fields.delta_scale.astype(np.float16),
            offset_norm=v5.local_fields.offset_norm.astype(np.float16),
            confidence=v5.local_fields.confidence.astype(np.float16),
            depth_low=v5.local_fields.depth_low.astype(np.float16),
            photo_range=np.float32(v5.local_fields.photo_range),
        )
    np.savez_compressed(
        result_dir / "temporal_errors.npz",
        v1=v1_temporal,
        v4=v3_temporal,
        **({"v5": v5_temporal} if v5_temporal is not None else {}),
    )
    np.savez_compressed(
        result_dir / "v4_parameters.npz",
        scales=v3.scales,
        offsets=v3.offsets,
        confidences=v3.confidences,
        lut_x=v3.lut_x,
        lut_y=v3.lut_y,
        residual_grids=v3.residual_grids,
        static_mask=v3.static_mask,
        static_target_grid=v3.static_target_grid,
        region_labels=v3.region_labels,
        region_scales=v3.region_scales,
        region_offsets=v3.region_offsets,
        region_shifts=v3.region_shifts,
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
    if metrics and "v5_anchor_nmae" in metrics[0]:
        lines = [
            "# DepthSync V5 验证结果",
            "",
            "V5 使用照片锚帧拟合低分辨率局部 affine 场，再通过双向光流向前后帧纯传输。下表中的时序指标均经过运动补偿；数值越低越好。",
            "",
            "| 场景 | 锚帧 NMAE V4→V5 | 主体锚帧 NMAE V4→V5 | 全局时序 P95 V4→V5 | 主体时序 P95 V4→V5 | V5 最大新增跳变 | 参数字节 | Python 回放 ms/帧 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in metrics:
            lines.append(
                f"| {item['scene']} | {item['v4_anchor_nmae']:.5f}→{item['v5_anchor_nmae']:.5f} | "
                f"{item['v4_subject_anchor_nmae']:.5f}→{item['v5_subject_anchor_nmae']:.5f} | "
                f"{item['v4_temporal_p95']:.5f}→{item['v5_temporal_p95']:.5f} | "
                f"{item['v4_subject_temporal_p95']:.5f}→{item['v5_subject_temporal_p95']:.5f} | "
                f"{item['v5_excess_jump_max']:.6f}（帧 {item['v5_excess_jump_frame']}） | "
                f"{item['v5_parameter_bytes']} | {item['v5_apply_ms_per_frame']:.2f} |"
            )
        lines.extend(["", "## 验收门槛", ""])
        for item in metrics:
            gates = item.get("v5_gates", {})
            if isinstance(gates, dict):
                for name, gate in gates.items():
                    if isinstance(gate, dict):
                        state = "通过" if gate.get("passed") else "失败"
                        lines.append(
                            f"- {item['scene']} / {name}: **{state}**；"
                            f"actual={gate.get('actual')}, limit={gate.get('limit')}"
                        )
        lines.extend(
            [
                "",
                "## 说明",
                "",
                "- 01 墙面门槛允许相对 V4 最多 10% 的平坦区修正梯度增量；这是为局部尺度对齐保留的有限空间变化，不允许出现硬分层。关键帧仍需人工检查。",
                "- V5 参数按三通道 FP16、90 帧、36×64 网格计算，共 1,244,160 字节；`depth_low` 是原型诊断缓存，不属于下发参数。",
                "- Python 回放耗时包含深度引导局部场上采样，仅供相对比较。端侧应以 C++/NEON、GPU 或 NPU kernel 实测，不能直接用该数值推断手机性能。",
                "- SEA-RAFT-S 的桌面 CPU 实测约 35–41 秒/90 帧；1–2 秒准备目标必须在目标手机 NPU/GPU 上完成 profiling 后才能确认。",
                "- 对比视频统一使用 DepthPro 照片深度的 P02–P98 显示范围，避免逐帧自动拉伸掩盖尺度差异。",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines = [
        "# DepthSync V4 全局单调 LUT 验证结果",
        "",
        "| 场景 | 锚帧 NMAE V1→V4 | 人脸切换 V1→V4 | 全局切换 V1→V4 | 全局时序 P95 V1→V4 | 主体时序 P95 V1→V4 | 最大新增跳变（帧） | V4 ms/帧 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        lines.append(
            f"| {item['scene']} | {item['v1_anchor_nmae']:.4f} → {item['v4_anchor_nmae']:.4f} | "
            f"{item['v1_switch_face']:.4f} → {item['v4_switch_face']:.4f} | "
            f"{item['v1_switch_global']:.4f} → {item['v4_switch_global']:.4f} | "
            f"{item['v1_temporal_p95']:.4f} → {item['v4_temporal_p95']:.4f} | "
            f"{item['v1_subject_temporal_p95']:.4f} → {item['v4_subject_temporal_p95']:.4f} | "
            f"{item['v4_excess_jump_max']:.4f}（{item['v4_excess_jump_frame']}） | "
            f"{item['v4_apply_ms_per_frame']:.3f} |"
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
            "- V3.6 定位到墙边约 15 像素宽的回退带：静态照片层在深度边缘被关闭后重新露出尺度不同的视频基础深度。新版用扩展人脸区域内的近景先验保护人物，同时允许静态背景掩码完成到物体边界。",
            "- V3.7 修复 V3.6 的主体副作用：短时主体占用由 3 帧时域中值 + 全范围检测捕获；人物运动走廊不再形成轮廓形静态掩码孔洞，而是污染其所在的完整照片深度连通平面。主体走廊关闭 16×9 空间残差，仅保留固定 LUT 和全片恒定的小 offset。",
            "- V4 最终撤下运行时空间区域修正：固定标签的全局平移会在 01 第 0 帧产生左右竖缝，常量区域 offset 会在第 60 帧形成“7”形硬轮廓。默认路径改为自适应选择 8/12/16 有效节点、统一序列化为 16 节点的全局单调 LUT，从机制上不再产生空间接缝。",
            "",
            "## 01 左墙与空间接缝专项检查",
            "",
            f"- V1 墙面中位值全片范围：{metrics[0].get('v1_static_median_range', float('nan')):.4f}。",
            f"- V4 墙面中位值全片范围：{metrics[0].get('v4_static_median_range', float('nan')):.4f}；纯全局映射不再单独锁定墙面，这是取消空间接缝后的明确取舍。",
            "- V4 默认区域标签为空，不执行标签 warp 或区域 offset；第 0 帧左右竖条/黑缝和第 60 帧“7”形轮廓没有生成路径。",
            "",
            "## 口径",
            "",
            "- V1 是逐帧全局 affine；V4 在锚帧自适应选择 8/12/16 有效节点的单调 LUT，随后统一为 16 节点定长结构，整段固定非线性形状。",
            "- 照片深度只在离线准备阶段参与参数估计；播放阶段只执行全局 LUT 和逐帧小 affine，不读取照片深度、不应用任何空间参数，也不需要人物分割 mask。",
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
    parser.add_argument("--flow-root", type=Path, default=Path("artifacts/flow"))
    parser.add_argument("--result-root", type=Path, default=Path("results"))
    parser.add_argument("--algorithm-version", choices=("v4", "v5"), default="v4")
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
            flow_root=args.flow_root,
            algorithm_version=args.algorithm_version,
            anchor_model=args.anchor_model,
        )
        for scene in args.scenes
    ]
    write_report(metrics, args.report)


if __name__ == "__main__":
    main()
