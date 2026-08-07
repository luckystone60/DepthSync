from __future__ import annotations

import numpy as np

from depthsync.evaluate import (
    classify_v5_gates,
    flat_correction_gradient_p99,
    temporal_frame_indices,
)


def test_flat_correction_gradient_ignores_real_depth_edges() -> None:
    """Counting a correction jump aligned to a real edge must make this fail."""
    base = np.full((16, 16), 0.3, np.float32)
    base[:, 8:] = 0.8
    synced = base + 0.01
    synced[:, 8:] += 0.5

    value = flat_correction_gradient_p99(synced, base, flat_threshold=0.02)

    assert value < 1e-6


def test_scene_02_gate_rejects_subject_temporal_regression() -> None:
    """Relaxing the 5% subject temporal limit must make this fail."""
    metrics = {
        "v4_subject_temporal_p95": 0.020,
        "v5_subject_temporal_p95": 0.020 * 1.051,
        "v5_revealed_background_confidence_p95": 0.02,
        "v5_parameter_bytes": 1_244_160,
    }

    gates = classify_v5_gates("02", metrics)

    assert gates["02_subject_temporal"]["passed"] is False


def test_scene_01_gate_requires_twenty_percent_subject_improvement() -> None:
    """Accepting less than 20% local anchor improvement must make this fail."""
    metrics = {
        "v4_subject_anchor_nmae": 0.10,
        "v5_subject_anchor_nmae": 0.081,
        "v4_wall_flat_correction_gradient_p99": 0.01,
        "v5_wall_flat_correction_gradient_p99": 0.01,
        "v5_parameter_bytes": 1_244_160,
    }

    gates = classify_v5_gates("01", metrics)

    assert gates["01_subject_anchor_nmae"]["passed"] is False


def test_scene_01_wall_gate_allows_small_smooth_correction_cost() -> None:
    """A brittle no-increase wall gate must make this fail."""
    metrics = {
        "v4_subject_anchor_nmae": 0.10,
        "v5_subject_anchor_nmae": 0.05,
        "v4_wall_flat_correction_gradient_p99": 0.0010,
        "v5_wall_flat_correction_gradient_p99": 0.00106,
        "v5_parameter_bytes": 1_244_160,
    }

    gates = classify_v5_gates("01", metrics)

    assert gates["01_wall_flat_correction_gradient_p99"]["passed"] is True


def test_new_scenes_use_generic_anchor_and_temporal_gates() -> None:
    metrics = {
        "v4_anchor_nmae": 0.10,
        "v5_anchor_nmae": 0.08,
        "v4_subject_anchor_nmae": 0.12,
        "v5_subject_anchor_nmae": 0.09,
        "v4_temporal_p95": 0.01,
        "v5_temporal_p95": 0.0114,
        "v5_parameter_bytes": 1_244_160,
    }

    gates = classify_v5_gates("20", metrics)

    assert gates["generic_anchor_nmae"]["passed"] is True
    assert gates["generic_subject_anchor_nmae"]["passed"] is True
    assert gates["generic_temporal_p95"]["passed"] is False


def test_tail_frame_config_maps_to_temporal_error_indices() -> None:
    indices = temporal_frame_indices([0, 1, 75, 89, 100], frame_count=90)

    np.testing.assert_array_equal(indices, [0, 74, 88])
