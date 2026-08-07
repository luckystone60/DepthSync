from __future__ import annotations

import numpy as np
import pytest

from depthsync.flow import (
    DenseFlowSequence,
    compute_flow_confidence,
    detect_scene_cuts,
    load_flow,
    save_flow,
)


def test_forward_backward_error_rejects_inconsistent_source_strip() -> None:
    """Dropping the forward/backward consistency term must make this fail."""
    forward = np.zeros((8, 12, 2), np.float32)
    backward = np.zeros_like(forward)
    forward[..., 0] = 2.0
    backward[..., 0] = -2.0
    # Forward pixels at x=5:7 sample backward pixels at x=7:9.
    backward[:, 7:9, 0] = 0.0

    confidence = compute_flow_confidence(
        forward,
        backward,
        np.ones((8, 12), np.float32),
    )

    assert np.median(confidence[:, 5:7]) < 0.2
    assert np.median(confidence[:, :3]) > 0.8
    assert not np.any(confidence[:, 10:])


def test_scene_cut_blocks_the_changed_pair_only() -> None:
    """Using a sequence-wide cut flag must make this fail."""
    black = np.zeros((16, 24, 3), np.uint8)
    white = np.full_like(black, 255)

    cuts = detect_scene_cuts([black, black, white, white])

    np.testing.assert_array_equal(cuts, [False, True, False])


def test_dense_flow_round_trip_preserves_direction_and_shape(tmp_path) -> None:
    """Swapping flow directions or frame metadata must make this fail."""
    forward = np.zeros((2, 4, 6, 2), np.float32)
    backward = np.zeros_like(forward)
    forward[..., 0] = 1.25
    backward[..., 0] = -1.25
    confidence = np.full((2, 4, 6), 0.75, np.float32)
    sequence = DenseFlowSequence(
        to_next=forward,
        to_previous=backward,
        confidence_next=confidence,
        confidence_previous=confidence,
        scene_cuts=np.asarray([False, True]),
        frame_shape=(8, 12),
    )
    path = tmp_path / "flow.npz"

    save_flow(path, sequence)
    loaded = load_flow(path)

    np.testing.assert_array_equal(loaded.to_next, forward.astype(np.float16))
    np.testing.assert_array_equal(loaded.to_previous, backward.astype(np.float16))
    np.testing.assert_array_equal(loaded.confidence_next, confidence.astype(np.float16))
    np.testing.assert_array_equal(loaded.scene_cuts, [False, True])
    assert loaded.frame_shape == (8, 12)


def test_scene_cut_zeroes_confidence_in_both_directions() -> None:
    """Allowing either direction to cross a cut must make this fail."""
    pair_shape = (2, 3, 4)
    sequence = DenseFlowSequence(
        to_next=np.zeros(pair_shape + (2,), np.float32),
        to_previous=np.zeros(pair_shape + (2,), np.float32),
        confidence_next=np.ones(pair_shape, np.float32),
        confidence_previous=np.ones(pair_shape, np.float32),
        scene_cuts=np.asarray([False, True]),
        frame_shape=(6, 8),
    )

    isolated = sequence.isolate_scene_cuts()

    np.testing.assert_array_equal(isolated.confidence_next[0], 1.0)
    np.testing.assert_array_equal(isolated.confidence_previous[0], 1.0)
    np.testing.assert_array_equal(isolated.confidence_next[1], 0.0)
    np.testing.assert_array_equal(isolated.confidence_previous[1], 0.0)


def test_dense_flow_rejects_non_finite_values() -> None:
    pair_shape = (1, 3, 4)
    forward = np.zeros(pair_shape + (2,), np.float32)
    forward[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        DenseFlowSequence(
            to_next=forward,
            to_previous=np.zeros_like(forward),
            confidence_next=np.ones(pair_shape, np.float32),
            confidence_previous=np.ones(pair_shape, np.float32),
            scene_cuts=np.zeros(1, bool),
            frame_shape=(3, 4),
        )


def test_dense_flow_rejects_out_of_range_confidence() -> None:
    pair_shape = (1, 3, 4)
    with pytest.raises(ValueError, match="confidence"):
        DenseFlowSequence(
            to_next=np.zeros(pair_shape + (2,), np.float32),
            to_previous=np.zeros(pair_shape + (2,), np.float32),
            confidence_next=np.full(pair_shape, -0.01, np.float32),
            confidence_previous=np.ones(pair_shape, np.float32),
            scene_cuts=np.zeros(1, bool),
            frame_shape=(3, 4),
        )
