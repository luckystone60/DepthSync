from __future__ import annotations

import numpy as np

from depthsync.local_field import (
    LocalFieldConfig,
    LocalFieldSequence,
    apply_local_field,
    fit_anchor_field,
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


def test_local_fit_separates_regions_that_share_the_same_base_value() -> None:
    """Replacing local fits with one global offset must make this fail."""
    h, w = 72, 128
    base = np.full((h, w), 0.45, np.float32)
    photo = base.copy()
    photo[:, : w // 2] += 0.12
    photo[:, w // 2 :] -= 0.08
    config = LocalFieldConfig(grid_shape=(18, 32))

    field = fit_anchor_field(base, photo, config)
    output = apply_local_field(base, field, config)
    global_output = base + np.median(photo - base)

    assert np.mean(np.abs(output - photo)) < 0.5 * np.mean(
        np.abs(global_output - photo)
    )


def test_invalid_anchor_windows_stay_exact_identity() -> None:
    """Filling unsupported windows from neighbors must make this fail."""
    base = np.full((24, 32), np.nan, np.float32)
    photo = np.ones((24, 32), np.float32)

    field = fit_anchor_field(
        base,
        photo,
        LocalFieldConfig(grid_shape=(6, 8)),
    )

    assert not np.any(field.delta_scale)
    assert not np.any(field.offset_norm)
    assert not np.any(field.confidence)
