from __future__ import annotations

import numpy as np

from depthsync.evaluate import (
    temporal_frame_indices,
)


def test_tail_frame_config_maps_to_temporal_error_indices() -> None:
    indices = temporal_frame_indices([0, 1, 75, 89, 100], frame_count=90)

    np.testing.assert_array_equal(indices, [0, 74, 88])
