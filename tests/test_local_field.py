from __future__ import annotations

import numpy as np

from depthsync.local_field import (
    LocalFieldConfig,
    LocalFieldSequence,
    apply_local_field,
    load_local_fields,
    save_local_fields,
)


def test_zero_confidence_is_bit_exact_identity() -> None:
    """Removing the identity fast path must make this fail."""
    base = np.linspace(0.1, 0.9, 35, dtype=np.float32).reshape(5, 7)
    shape = (1, 3, 4)
    fields = LocalFieldSequence(
        delta_scale=np.full(shape, 0.4, np.float32),
        offset_norm=np.full(shape, 0.2, np.float32),
        confidence=np.zeros(shape, np.float32),
        depth_low=np.zeros(shape, np.float32),
        photo_range=0.8,
    )

    actual = apply_local_field(
        base,
        fields.frame(0),
        LocalFieldConfig(grid_shape=(3, 4)),
    )

    np.testing.assert_array_equal(actual, base)


def test_field_round_trip_preserves_fp16_payload(tmp_path) -> None:
    """Changing persisted channel dtype or values must make this fail."""
    shape = (3, 3, 4)
    count = int(np.prod(shape))
    fields = LocalFieldSequence(
        delta_scale=np.linspace(-0.1, 0.1, count, dtype=np.float32).reshape(shape),
        offset_norm=np.linspace(-0.2, 0.2, count, dtype=np.float32).reshape(shape),
        confidence=np.linspace(0.0, 1.0, count, dtype=np.float32).reshape(shape),
        depth_low=np.full(shape, 0.5, np.float32),
        photo_range=0.8,
    )
    path = tmp_path / "v5_fields.npz"

    save_local_fields(path, fields)
    loaded = load_local_fields(path)

    np.testing.assert_array_equal(
        loaded.delta_scale,
        fields.delta_scale.astype(np.float16),
    )
    np.testing.assert_array_equal(
        loaded.offset_norm,
        fields.offset_norm.astype(np.float16),
    )
    np.testing.assert_array_equal(
        loaded.confidence,
        fields.confidence.astype(np.float16),
    )
    np.testing.assert_array_equal(
        loaded.depth_low,
        fields.depth_low.astype(np.float16),
    )
    assert loaded.photo_range == fields.photo_range
