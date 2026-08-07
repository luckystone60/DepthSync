from __future__ import annotations

import numpy as np
import warnings

from depthsync.sea_raft import (
    SeaRaftEstimator,
    mixture_log_variance,
    validate_checkpoint_keys,
)


class DirectionalFakeBackend:
    """Fast stand-in for only the external neural inference boundary."""

    def predict(self, image1: np.ndarray, image2: np.ndarray):
        direction = np.sign(float(np.mean(image2) - np.mean(image1)))
        height, width = image1.shape[:2]
        flow = np.zeros((height, width, 2), np.float32)
        flow[..., 0] = 2.0 * direction
        log_variance = np.zeros((height, width), np.float32)
        return flow, log_variance


class MeshgridWarningBackend(DirectionalFakeBackend):
    def predict(self, image1: np.ndarray, image2: np.ndarray):
        warnings.warn(
            "torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.",
            UserWarning,
        )
        return super().predict(image1, image2)


def test_adapter_builds_opposite_forward_and_reverse_pairs() -> None:
    """Dropping reverse inference or swapping directions must make this fail."""
    estimator = SeaRaftEstimator.from_backend(
        DirectionalFakeBackend(),
        input_size=(16, 24),
    )
    frames = [
        np.full((32, 48, 3), value, np.uint8)
        for value in (0, 10, 20, 30)
    ]

    result = estimator.estimate_pairs(frames)

    assert result.to_next.shape == (3, 16, 24, 2)
    assert result.to_previous.shape == (3, 16, 24, 2)
    np.testing.assert_allclose(result.to_next[..., 0], 2.0)
    np.testing.assert_allclose(result.to_previous[..., 0], -2.0)
    assert float(np.median(result.confidence_next[:, :, :20])) > 0.5
    assert result.frame_shape == (32, 48)


def test_mixture_uncertainty_uses_softmax_weighted_clamped_log_scales() -> None:
    """Using raw logits or unclamped scales must make this fail."""
    info = np.zeros((4, 2, 3), np.float32)
    info[2] = 2.0
    info[3] = -2.0

    actual = mixture_log_variance(info, var_min=-1.0, var_max=1.0)

    np.testing.assert_allclose(actual, 0.0, atol=1e-6)


def test_official_checkpoint_allows_only_unstored_batchnorm_buffers() -> None:
    """Rejecting the official sparse buffers or accepting arbitrary keys must fail."""
    validate_checkpoint_keys(
        ["cnet.layer2.0.downsample.1.running_mean"],
        [],
    )
    try:
        validate_checkpoint_keys([], ["unknown.weight"])
    except RuntimeError as error:
        assert "unexpected" in str(error)
    else:
        raise AssertionError("unexpected checkpoint keys were accepted")


def test_adapter_suppresses_only_the_known_upstream_meshgrid_warning() -> None:
    """Leaking the pinned upstream warning into every validation run must fail."""
    estimator = SeaRaftEstimator.from_backend(
        MeshgridWarningBackend(), input_size=(8, 8)
    )
    frames = [
        np.zeros((8, 8, 3), np.uint8),
        np.ones((8, 8, 3), np.uint8),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.estimate_pairs(frames)
    assert caught == []
