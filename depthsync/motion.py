"""Validation adapter that approximates endpoint block motion with sparse LK."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .core import MotionSequence


def _read_gray_frames(video: Path, max_side: int = 640) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if max(height, width) > max_side:
                scale = max_side / max(height, width)
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    return frames


def _track(source: np.ndarray, target: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tracked, status, error = cv2.calcOpticalFlowPyrLK(
        source,
        target,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02),
    )
    if tracked is None:
        return np.zeros_like(points), np.zeros(len(points), np.float32)
    displacement = tracked - points
    status = status.reshape(-1).astype(np.float32)
    error = np.nan_to_num(error.reshape(-1), nan=255.0, posinf=255.0)
    confidence = status * np.exp(-error / 24.0)
    return displacement, confidence.astype(np.float32)


def estimate_block_motion(video: Path, grid_shape: tuple[int, int] = (18, 32), max_side: int = 640) -> MotionSequence:
    """Estimate bidirectional normalized motion on a small fixed grid."""
    frames = _read_gray_frames(video, max_side)
    if len(frames) < 2:
        raise ValueError(f"Need at least two frames for motion: {video}")
    height, width = frames[0].shape
    gh, gw = grid_shape
    xs = np.linspace(0, width - 1, gw, dtype=np.float32)
    ys = np.linspace(0, height - 1, gh, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
    to_previous = np.zeros((len(frames), gh, gw, 2), np.float32)
    to_next = np.zeros_like(to_previous)
    confidence_previous = np.zeros((len(frames), gh, gw), np.float32)
    confidence_next = np.zeros_like(confidence_previous)
    for index in range(len(frames) - 1):
        forward, forward_conf = _track(frames[index], frames[index + 1], points)
        backward, backward_conf = _track(frames[index + 1], frames[index], points)
        forward[..., 0] /= max(width - 1, 1)
        forward[..., 1] /= max(height - 1, 1)
        backward[..., 0] /= max(width - 1, 1)
        backward[..., 1] /= max(height - 1, 1)
        to_next[index] = forward.reshape(gh, gw, 2)
        confidence_next[index] = forward_conf.reshape(gh, gw)
        to_previous[index + 1] = backward.reshape(gh, gw, 2)
        confidence_previous[index + 1] = backward_conf.reshape(gh, gw)
    return MotionSequence(to_previous, to_next, confidence_previous, confidence_next)


def save_motion(path: Path, motion: MotionSequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        to_previous=motion.to_previous,
        to_next=motion.to_next,
        confidence_previous=motion.confidence_previous,
        confidence_next=motion.confidence_next,
    )


def load_motion(path: Path) -> MotionSequence:
    data = np.load(path)
    return MotionSequence(data["to_previous"], data["to_next"], data["confidence_previous"], data["confidence_next"])
