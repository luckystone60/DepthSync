"""Optional SEA-RAFT-S adapter for the post-capture preparation stage."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import warnings
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from .flow import DenseFlowSequence, compute_flow_confidence, detect_scene_cuts


class FlowBackend(Protocol):
    def predict(self, image1: np.ndarray, image2: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class SeaRaftEstimator:
    """Generate cached adjacent flow without importing PyTorch at package load."""

    def __init__(
        self,
        checkpoint: Path | str,
        repository: Path | str,
        device: str = "cpu",
        input_size: tuple[int, int] = (144, 256),
    ):
        self._backend: FlowBackend = _OfficialSeaRaftBackend(Path(checkpoint), Path(repository), device)
        self.input_size = input_size

    @classmethod
    def from_backend(
        cls, backend: FlowBackend, input_size: tuple[int, int] = (144, 256)
    ) -> "SeaRaftEstimator":
        estimator = cls.__new__(cls)
        estimator._backend = backend
        estimator.input_size = input_size
        return estimator

    def estimate_pairs(self, frames: Sequence[np.ndarray]) -> DenseFlowSequence:
        if len(frames) < 2:
            raise ValueError("at least two RGB frames are required")
        first_shape = np.asarray(frames[0]).shape[:2]
        input_h, input_w = self.input_size
        prepared: list[np.ndarray] = []
        gray: list[np.ndarray] = []
        for frame in frames:
            image = np.asarray(frame)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("frames must be BGR images")
            resized = cv2.resize(image, (input_w, input_h), cv2.INTER_AREA)
            prepared.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            gray.append(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32))
        to_next, to_previous = [], []
        confidence_next, confidence_previous = [], []
        for index in range(len(prepared) - 1):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"torch\.meshgrid: in an upcoming release.*", category=UserWarning)
                forward, forward_log_variance = self._backend.predict(prepared[index], prepared[index + 1])
                backward, backward_log_variance = self._backend.predict(prepared[index + 1], prepared[index])
            forward = _validate_prediction(forward, (input_h, input_w), "forward")
            backward = _validate_prediction(backward, (input_h, input_w), "backward")
            to_next.append(forward)
            to_previous.append(backward)
            confidence_next.append(
                compute_flow_confidence(
                    forward, backward, _variance_confidence(forward_log_variance, (input_h, input_w)),
                    gray[index], gray[index + 1],
                )
            )
            confidence_previous.append(
                compute_flow_confidence(
                    backward, forward, _variance_confidence(backward_log_variance, (input_h, input_w)),
                    gray[index + 1], gray[index],
                )
            )
        return DenseFlowSequence(
            np.stack(to_next),
            np.stack(to_previous),
            np.stack(confidence_next),
            np.stack(confidence_previous),
            detect_scene_cuts(frames),
            (int(first_shape[0]), int(first_shape[1])),
        ).isolate_scene_cuts()


def _validate_prediction(flow: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    value = np.asarray(flow, np.float32)
    if value.shape != shape + (2,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} flow must be finite with shape [H,W,2]")
    return value


def _variance_confidence(log_variance: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(log_variance, np.float32)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError("log variance must be finite with shape [H,W]")
    return np.exp(-0.5 * np.clip(value, 0.0, 10.0)).astype(np.float32)


def mixture_log_variance(info: np.ndarray, var_min: float, var_max: float) -> np.ndarray:
    """Collapse SEA-RAFT's two-component Laplace uncertainty to one map."""
    value = np.asarray(info, np.float32)
    if value.ndim != 3 or value.shape[0] != 4:
        raise ValueError("SEA-RAFT info must have shape [4,H,W]")
    logits = value[:2] - np.max(value[:2], axis=0, keepdims=True)
    weights = np.exp(logits)
    weights /= np.maximum(np.sum(weights, axis=0, keepdims=True), 1e-12)
    scales = np.empty_like(value[2:])
    scales[0] = np.clip(value[2], 0.0, var_max)
    scales[1] = np.clip(value[3], var_min, 0.0)
    return np.sum(weights * scales, axis=0).astype(np.float32)


def validate_checkpoint_keys(missing: Sequence[str], unexpected: Sequence[str]) -> None:
    invalid = [
        key for key in missing
        if not (key.startswith(("cnet.", "fnet.")) and ".downsample.1." in key)
    ]
    if invalid or unexpected:
        raise RuntimeError(f"SEA-RAFT checkpoint mismatch: missing={invalid}, unexpected={list(unexpected)}")


class _OfficialSeaRaftBackend:
    def __init__(self, checkpoint: Path, repository: Path, device: str):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SEA-RAFT checkpoint not found: {checkpoint}")
        if not repository.is_dir():
            raise FileNotFoundError(f"SEA-RAFT repository not found: {repository}")
        self._checkpoint = checkpoint
        self._repository = repository
        self._device = device
        self._model = None
        self._torch = None
        self._args = None

    def predict(self, image1: np.ndarray, image2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            self._load_model()
        return self._predict_loaded(image1, image2)

    def _load_model(self) -> None:
        import torch
        from safetensors.torch import load_file

        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        config_path = self._repository / "config" / "eval" / "spring-S.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"SEA-RAFT config not found: {config_path}")
        args = argparse.Namespace(**json.loads(config_path.read_text(encoding="utf-8")))
        core_path = str((self._repository / "core").resolve())
        if core_path not in sys.path:
            sys.path.insert(0, core_path)
        model = importlib.import_module("raft").RAFT(args)
        state_dict = load_file(str(self._checkpoint), device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        validate_checkpoint_keys(missing, unexpected)
        self._torch = torch
        self._args = args
        self._model = model.to(torch.device(self._device)).eval()

    def _predict_loaded(self, image1: np.ndarray, image2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._torch is not None and self._args is not None
        torch = self._torch
        tensor1 = torch.from_numpy(np.ascontiguousarray(image1.transpose(2, 0, 1))).float()[None].to(self._device)
        tensor2 = torch.from_numpy(np.ascontiguousarray(image2.transpose(2, 0, 1))).float()[None].to(self._device)
        with torch.no_grad():
            output = self._model(tensor1, tensor2, iters=self._args.iters, test_mode=True)
        flow = output["flow"][-1][0].permute(1, 2, 0).cpu().numpy()
        info = output["info"][-1][0].cpu().numpy()
        log_variance = mixture_log_variance(info, float(self._args.var_min), float(self._args.var_max))
        return flow.astype(np.float32), log_variance
