"""Dense adjacent-flow contracts used by the offline local alignment stage."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class DenseFlowSequence:
    """Bidirectional adjacent flow in stored-grid pixel units.

    ``to_next[p]`` maps frame ``p`` to ``p + 1`` and ``to_previous[p]`` maps
    frame ``p + 1`` to ``p``.  Both arrays therefore contain ``T - 1`` pairs.
    """

    to_next: np.ndarray
    to_previous: np.ndarray
    confidence_next: np.ndarray
    confidence_previous: np.ndarray
    scene_cuts: np.ndarray
    frame_shape: tuple[int, int]

    def __post_init__(self) -> None:
        shape = self.to_next.shape
        if len(shape) != 4 or shape[-1] != 2:
            raise ValueError("dense flow must have shape [T-1,H,W,2]")
        if self.to_previous.shape != shape:
            raise ValueError("forward and backward flow shapes must match")
        if self.confidence_next.shape != shape[:-1] or self.confidence_previous.shape != shape[:-1]:
            raise ValueError("flow confidence must have shape [T-1,H,W]")
        if self.scene_cuts.shape != (shape[0],):
            raise ValueError("scene_cuts must have one value per pair")
        if len(self.frame_shape) != 2 or min(self.frame_shape) <= 0:
            raise ValueError("frame_shape must contain positive height and width")
        for values in (self.to_next, self.to_previous, self.confidence_next, self.confidence_previous):
            if not np.all(np.isfinite(values)):
                raise ValueError("flow payload must be finite")
        for confidence in (self.confidence_next, self.confidence_previous):
            if np.any((confidence < 0.0) | (confidence > 1.0)):
                raise ValueError("flow confidence must be within [0,1]")

    @property
    def frame_count(self) -> int:
        return int(self.to_next.shape[0] + 1)

    def isolate_scene_cuts(self) -> "DenseFlowSequence":
        active = (~self.scene_cuts).astype(np.float32)[:, None, None]
        return DenseFlowSequence(
            self.to_next,
            self.to_previous,
            (self.confidence_next * active).astype(np.float32),
            (self.confidence_previous * active).astype(np.float32),
            self.scene_cuts,
            self.frame_shape,
        )


def load_dense_flow(path: Path | str) -> DenseFlowSequence:
    """Load a cached small-flow result without retaining an open NPZ handle."""
    with np.load(Path(path), allow_pickle=False) as payload:
        return DenseFlowSequence(
            to_next=payload["to_next"].astype(np.float32, copy=True),
            to_previous=payload["to_previous"].astype(np.float32, copy=True),
            confidence_next=payload["confidence_next"].astype(np.float32, copy=True),
            confidence_previous=payload["confidence_previous"].astype(np.float32, copy=True),
            scene_cuts=payload["scene_cuts"].astype(bool, copy=True),
            frame_shape=tuple(int(value) for value in payload["frame_shape"]),
        )


def save_dense_flow(path: Path | str, flow: DenseFlowSequence) -> None:
    np.savez_compressed(
        Path(path),
        to_next=flow.to_next.astype(np.float16),
        to_previous=flow.to_previous.astype(np.float16),
        confidence_next=flow.confidence_next.astype(np.float16),
        confidence_previous=flow.confidence_previous.astype(np.float16),
        scene_cuts=flow.scene_cuts.astype(bool),
        frame_shape=np.asarray(flow.frame_shape, np.int32),
    )


def compute_flow_confidence(
    forward: np.ndarray,
    backward: np.ndarray,
    model_confidence: np.ndarray,
    source_gray: np.ndarray | None = None,
    target_gray: np.ndarray | None = None,
) -> np.ndarray:
    """Combine model uncertainty, forward/backward consistency and appearance."""
    forward = np.asarray(forward, np.float32)
    backward = np.asarray(backward, np.float32)
    model_confidence = np.asarray(model_confidence, np.float32)
    if forward.ndim != 3 or forward.shape[-1] != 2 or backward.shape != forward.shape:
        raise ValueError("forward and backward must have equal [H,W,2] shapes")
    if model_confidence.shape != forward.shape[:2]:
        raise ValueError("model_confidence must have shape [H,W]")
    height, width = forward.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x, map_y = xx + forward[..., 0], yy + forward[..., 1]
    in_bounds = (map_x >= 0.0) & (map_x <= width - 1) & (map_y >= 0.0) & (map_y <= height - 1)
    reverse = cv2.remap(
        backward, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    error = np.linalg.norm(forward + reverse, axis=-1)
    tolerance = 1.5 + 0.01 * np.linalg.norm(forward, axis=-1)
    confidence = np.clip(model_confidence, 0.0, 1.0) * np.exp(-np.square(error / tolerance))
    if source_gray is not None and target_gray is not None:
        source = np.asarray(source_gray, np.float32)
        target = np.asarray(target_gray, np.float32)
        if source.shape != (height, width) or target.shape != (height, width):
            raise ValueError("photometric frames must match flow resolution")
        target_warped = cv2.remap(target, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        brightness_error = np.minimum(np.abs(source - target_warped) / 64.0, 1.0)
        confidence *= np.exp(-brightness_error)
    confidence *= in_bounds.astype(np.float32)
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def detect_scene_cuts(frames: Sequence[np.ndarray], threshold: float = 0.35) -> np.ndarray:
    if len(frames) < 2:
        return np.zeros(0, bool)
    gray = []
    for frame in frames:
        image = np.asarray(frame)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim != 2:
            raise ValueError("frames must contain grayscale or BGR images")
        gray.append(cv2.resize(image.astype(np.float32), (32, 18), cv2.INTER_AREA))
    scores = [float(np.median(np.abs(current - previous)) / 255.0) for previous, current in zip(gray, gray[1:])]
    return np.asarray(scores, np.float32) >= threshold


def _remap(values: np.ndarray, coordinates: np.ndarray, border_value: float = 0.0) -> np.ndarray:
    return cv2.remap(
        values.astype(np.float32),
        coordinates[..., 0],
        coordinates[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def compose_to_anchor(
    flow: DenseFlowSequence,
    anchor_index: int,
    confidence_floor: float = 0.05,
    distance_half_life: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose adjacent flow into normalized current-to-anchor coordinates.

    Confidence uses the weakest link along a track plus a gentle distance
    decay.  Unlike multiplying all pair confidences, this remains usable over
    a 90-frame Live Photo while still rejecting any unreliable or cut pair.
    """
    if not 0 <= anchor_index < flow.frame_count:
        raise ValueError("anchor_index is out of range")
    pair_count, height, width = flow.to_next.shape[:3]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    identity = np.stack((xx, yy), axis=-1)
    coordinates = np.zeros((pair_count + 1, height, width, 2), np.float32)
    confidence = np.zeros((pair_count + 1, height, width), np.float32)
    coordinates[anchor_index] = identity
    confidence[anchor_index] = 1.0

    for direction in (-1, 1):
        stop = -1 if direction < 0 else pair_count + 1
        for index in range(anchor_index + direction, stop, direction):
            if direction < 0:
                pair = index
                step = flow.to_next[pair]
                step_confidence = flow.confidence_next[pair]
                nearer = index + 1
            else:
                pair = index - 1
                step = flow.to_previous[pair]
                step_confidence = flow.confidence_previous[pair]
                nearer = index - 1
            current_to_nearer = identity + step
            coordinates[index] = _remap(coordinates[nearer], current_to_nearer)
            inherited = _remap(confidence[nearer], current_to_nearer)
            in_bounds = (
                (current_to_nearer[..., 0] >= 0.0)
                & (current_to_nearer[..., 0] <= width - 1.0)
                & (current_to_nearer[..., 1] >= 0.0)
                & (current_to_nearer[..., 1] <= height - 1.0)
            )
            pair_confidence = np.clip(step_confidence, 0.0, 1.0)
            if flow.scene_cuts[pair]:
                pair_confidence = np.zeros_like(pair_confidence)
            track_confidence = np.minimum(inherited, pair_confidence)
            track_confidence *= in_bounds.astype(np.float32)
            distance = abs(index - anchor_index)
            track_confidence *= np.exp(-np.log(2.0) * distance / max(distance_half_life, 1.0))
            confidence[index] = np.where(
                track_confidence >= confidence_floor,
                track_confidence,
                0.0,
            )

    coordinates[..., 0] /= max(width - 1, 1)
    coordinates[..., 1] /= max(height - 1, 1)
    return coordinates, np.clip(confidence, 0.0, 1.0)
