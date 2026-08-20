from __future__ import annotations

import numpy as np

from depthsync.flow import DenseFlowSequence, compose_to_anchor
from depthsync.local_alignment import (
    LocalAlignmentConfig,
    LocalFieldFrame,
    apply_local_field,
    apply_local_sequence,
    fit_flow_guided_fields,
    select_temporal_safe_fields,
    LocalFieldSequence,
)
from depthsync.registration import estimate_photo_registration


def _translation_flow(frame_count: int, shape: tuple[int, int], pixels: float = 1.0) -> DenseFlowSequence:
    height, width = shape
    pair_shape = (frame_count - 1, height, width)
    to_next = np.zeros(pair_shape + (2,), np.float32)
    to_previous = np.zeros_like(to_next)
    to_next[..., 0] = pixels
    to_previous[..., 0] = -pixels
    confidence = np.ones(pair_shape, np.float32)
    return DenseFlowSequence(
        to_next,
        to_previous,
        confidence,
        confidence.copy(),
        np.zeros(frame_count - 1, bool),
        shape,
    )


def test_compose_to_anchor_tracks_translation_in_both_directions() -> None:
    coordinates, confidence = compose_to_anchor(_translation_flow(5, (12, 20)), 2)
    center = (6, 8)
    # Frame 0 needs two +1 steps to reach anchor; frame 4 needs two -1 steps.
    assert abs(float(coordinates[0, center[0], center[1], 0]) * 19 - 10.0) < 1e-4
    assert abs(float(coordinates[4, center[0], center[1], 0]) * 19 - 6.0) < 1e-4
    assert confidence[0, center[0], center[1]] > 0.9
    assert confidence[4, center[0], center[1]] > 0.9


def test_scene_cut_removes_all_support_beyond_cut() -> None:
    flow = _translation_flow(5, (8, 12), 0.0)
    flow.scene_cuts[2] = True
    _, confidence = compose_to_anchor(flow, 1)
    assert np.max(confidence[2]) > 0.0
    assert np.max(confidence[3:]) == 0.0


def test_zero_confidence_local_field_is_exact_identity() -> None:
    base = np.linspace(0.1, 0.9, 48, dtype=np.float32).reshape(6, 8)
    shape = (2, 3)
    frame = LocalFieldFrame(
        np.full(shape, 0.08, np.float32),
        np.full(shape, 0.1, np.float32),
        np.zeros(shape, np.float32),
        np.full(shape, 0.5, np.float32),
        1.0,
    )
    np.testing.assert_array_equal(apply_local_field(base, frame), base)


def test_guided_interpolation_does_not_cross_depth_edge() -> None:
    base = np.full((12, 40), 0.2, np.float32)
    base[:, 20:] = 0.8
    frame = LocalFieldFrame(
        np.zeros((1, 2), np.float32),
        np.asarray([[0.1, -0.1]], np.float32),
        np.ones((1, 2), np.float32),
        np.asarray([[0.2, 0.8]], np.float32),
        1.0,
    )
    output = apply_local_field(
        base,
        frame,
        LocalAlignmentConfig(grid_shape=(1, 2), depth_sigma_fraction=0.01),
    )
    assert float(np.median(output[:, :16] - base[:, :16])) > 0.08
    assert float(np.median(output[:, 24:] - base[:, 24:])) < -0.08


def test_flow_guided_fit_separates_two_local_offsets_without_hard_seam() -> None:
    frame_count, height, width, anchor = 5, 48, 72, 2
    plane = np.tile(np.linspace(0.2, 1.0, width, dtype=np.float32), (height, 1))
    base = np.repeat(plane[None], frame_count, axis=0)
    photo = plane.copy()
    photo[:, : width // 2] += 0.10
    photo[:, width // 2 :] -= 0.08
    flow = _translation_flow(frame_count, (24, 36), 0.0)
    config = LocalAlignmentConfig(
        grid_shape=(6, 9),
        min_fit_pixels=24,
        window_scale=2.0,
        min_improvement=0.05,
        offset_bound_fraction=0.25,
        correction_bound_fraction=0.25,
        temporal_transport_weight=0.5,
    )
    fields = fit_flow_guided_fields(base, photo, anchor, flow, config)
    output = apply_local_sequence(base, fields, config)
    before = float(np.mean(np.abs(base[anchor] - photo)))
    after = float(np.mean(np.abs(output[anchor] - photo)))
    assert after < before * 0.55
    assert np.max(np.abs(np.diff(output[anchor], axis=1))) < 0.20
    np.testing.assert_allclose(output[0], output[-1], atol=1e-5)


def test_registration_keeps_same_frame_identity() -> None:
    yy, xx = np.mgrid[:80, :120]
    image = np.stack(((xx * 3) % 255, (yy * 5) % 255, ((xx + yy) * 2) % 255), axis=-1).astype(np.uint8)
    photo = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
    registration = estimate_photo_registration(photo, image, image.shape[:2])
    assert registration.method == "identity"
    expected_x = np.linspace(0.0, 1.0, image.shape[1], dtype=np.float32)
    np.testing.assert_allclose(registration.grid[0, :, 0], expected_x, atol=1e-6)


def test_temporal_safety_gate_rejects_flickering_field() -> None:
    frame_count = 8
    base = np.repeat(
        np.tile(np.linspace(0.2, 1.0, 32, dtype=np.float32), (24, 1))[None],
        frame_count,
        axis=0,
    )
    shape = (frame_count, 2, 3)
    offset = np.zeros(shape, np.float32)
    offset[1::2] = 0.1
    offset[::2] = -0.1
    fields = LocalFieldSequence(
        np.zeros(shape, np.float32),
        offset,
        np.ones(shape, np.float32),
        np.full(shape, 0.6, np.float32),
        0.8,
    )
    config = LocalAlignmentConfig(
        grid_shape=(2, 3),
        temporal_p95_ratio=1.0,
        temporal_p95_budget=0.0,
        temporal_max_excess=0.001,
    )
    safe, strength, _ = select_temporal_safe_fields(
        base, fields, _translation_flow(frame_count, (12, 16), 0.0), config
    )
    assert strength == 0.0
    assert not np.any(safe.confidence)
