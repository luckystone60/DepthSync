from __future__ import annotations

import numpy as np

from depthsync.evaluate import (
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


def test_tail_frame_config_maps_to_temporal_error_indices() -> None:
    indices = temporal_frame_indices([0, 1, 75, 89, 100], frame_count=90)

    np.testing.assert_array_equal(indices, [0, 74, 88])
