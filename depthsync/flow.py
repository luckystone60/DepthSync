"""Dense bidirectional flow contracts and reliability for DepthSync V5."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class DenseFlowSequence:
    """Adjacent flow pairs in stored-flow pixel units.

    ``to_next[p]`` maps frame p to p+1. ``to_previous[p]`` maps frame
    p+1 to p, so both arrays contain T-1 entries.
    """

    to_next: np.ndarray
    to_previous: np.ndarray
    confidence_next: np.ndarray
    confidence_previous: np.ndarray
    scene_cuts: np.ndarray
    frame_shape: tuple[int, int]

    def __post_init__(self) -> None:
        pair_shape = self.to_next.shape
        if len(pair_shape) != 4 or pair_shape[-1] != 2:
            raise ValueError("dense flow must have shape [T-1,H,W,2]")
        if self.to_previous.shape != pair_shape:
            raise ValueError("forward and backward flow shapes must match")
        expected_confidence = pair_shape[:-1]
        if self.confidence_next.shape != expected_confidence or self.confidence_previous.shape != expected_confidence:
            raise ValueError("flow confidence must have shape [T-1,H,W]")
        if self.scene_cuts.shape != (pair_shape[0],):
            raise ValueError("scene_cuts must have one value per adjacent pair")
        if len(self.frame_shape) != 2 or min(self.frame_shape) <= 0:
            raise ValueError("frame_shape must contain positive height and width")

    def isolate_scene_cuts(self) -> "DenseFlowSequence":
        """Return a copy whose two direction confidences cannot cross cuts."""
        active = (~self.scene_cuts).astype(np.float32)[:, None, None]
        return DenseFlowSequence(
            to_next=self.to_next,
            to_previous=self.to_previous,
            confidence_next=(self.confidence_next * active).astype(np.float32),
            confidence_previous=(self.confidence_previous * active).astype(np.float32),
            scene_cuts=self.scene_cuts,
            frame_shape=self.frame_shape,
        )


def compute_flow_confidence(
    forward: np.ndarray,
    backward: np.ndarray,
    model_confidence: np.ndarray,
    source_gray: np.ndarray | None = None,
    target_gray: np.ndarray | None = None,
) -> np.ndarray:
    """Return continuous reliability from consistency, bounds and appearance."""
    forward = np.asarray(forward, dtype=np.float32)
    backward = np.asarray(backward, dtype=np.float32)
    model_confidence = np.asarray(model_confidence, dtype=np.float32)
    if forward.ndim != 3 or forward.shape[-1] != 2 or backward.shape != forward.shape:
        raise ValueError("forward and backward must have equal [H,W,2] shapes")
    if model_confidence.shape != forward.shape[:2]:
        raise ValueError("model_confidence must have shape [H,W]")
    height, width = forward.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x = xx + forward[..., 0]
    map_y = yy + forward[..., 1]
    in_bounds = (
        (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )
    backward_at_forward = cv2.remap(
        backward,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    fb_error = np.linalg.norm(forward + backward_at_forward, axis=-1)
    tau = 1.5 + 0.01 * np.linalg.norm(forward, axis=-1)
    confidence = np.clip(model_confidence, 0.0, 1.0) * np.exp(
        -np.square(fb_error / tau)
    )
    if source_gray is not None and target_gray is not None:
        source = np.asarray(source_gray, dtype=np.float32)
        target = np.asarray(target_gray, dtype=np.float32)
        if source.shape != (height, width) or target.shape != (height, width):
            raise ValueError("photometric frames must match the flow resolution")
        target_at_forward = cv2.remap(
            target,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        source_grad = cv2.magnitude(
            cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=3),
        )
        target_grad = cv2.magnitude(
            cv2.Sobel(target_at_forward, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(target_at_forward, cv2.CV_32F, 0, 1, ksize=3),
        )
        brightness_error = np.minimum(np.abs(source - target_at_forward) / 64.0, 1.0)
        gradient_error = np.minimum(np.abs(source_grad - target_grad) / 96.0, 1.0)
        confidence *= np.exp(-brightness_error - 0.5 * gradient_error)
    confidence *= in_bounds.astype(np.float32)
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def detect_scene_cuts(
    frames: Sequence[np.ndarray],
    threshold: float = 0.35,
) -> np.ndarray:
    """Detect adjacent hard cuts from robust low-resolution luminance change."""
    if len(frames) < 2:
        return np.zeros(0, bool)
    gray_frames: list[np.ndarray] = []
    for frame in frames:
        image = np.asarray(frame)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim != 2:
            raise ValueError("frames must contain grayscale or BGR images")
        gray_frames.append(
            cv2.resize(image.astype(np.float32), (32, 18), interpolation=cv2.INTER_AREA)
        )
    scores = [
        float(np.median(np.abs(current - previous)) / 255.0)
        for previous, current in zip(gray_frames, gray_frames[1:])
    ]
    return np.asarray(scores, np.float32) >= threshold


def save_flow(path: Path | str, flow: DenseFlowSequence) -> None:
    np.savez_compressed(
        Path(path),
        to_next=flow.to_next.astype(np.float16),
        to_previous=flow.to_previous.astype(np.float16),
        confidence_next=flow.confidence_next.astype(np.float16),
        confidence_previous=flow.confidence_previous.astype(np.float16),
        scene_cuts=flow.scene_cuts.astype(bool),
        frame_shape=np.asarray(flow.frame_shape, np.int32),
    )


def load_flow(path: Path | str) -> DenseFlowSequence:
    with np.load(Path(path), allow_pickle=False) as payload:
        frame_shape = tuple(int(value) for value in payload["frame_shape"])
        return DenseFlowSequence(
            to_next=payload["to_next"].astype(np.float16, copy=True),
            to_previous=payload["to_previous"].astype(np.float16, copy=True),
            confidence_next=payload["confidence_next"].astype(np.float16, copy=True),
            confidence_previous=payload["confidence_previous"].astype(np.float16, copy=True),
            scene_cuts=payload["scene_cuts"].astype(bool, copy=True),
            frame_shape=frame_shape,
        )
