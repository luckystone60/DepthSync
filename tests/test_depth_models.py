from __future__ import annotations

import json

import numpy as np
import pytest

from depthsync.dav2 import orient_relative_depth, write_dav2_outputs


def test_relative_depth_is_oriented_to_positive_anchor_correlation() -> None:
    relative = np.tile(np.linspace(0.0, 1.0, 8, dtype=np.float32), (8, 1))
    reference = 1.0 - relative

    disparity, reversed_direction = orient_relative_depth(relative, reference)

    np.testing.assert_allclose(disparity, reference)
    assert reversed_direction is True


def test_relative_depth_rejects_ambiguous_direction() -> None:
    relative = np.tile(np.linspace(0.0, 1.0, 8, dtype=np.float32), (8, 1))
    reference = np.ones_like(relative)

    with pytest.raises(ValueError, match="direction"):
        orient_relative_depth(relative, reference)


def test_dav2_outputs_record_direction_and_provenance(tmp_path) -> None:
    relative = np.full((8, 12), 0.4, np.float32)
    disparity = np.full((8, 12), 0.6, np.float32)
    metadata = {
        "backend": "Depth-Anything-V2-Large",
        "model_id": "depth-anything/Depth-Anything-V2-Large-hf",
        "revision": "test-revision",
        "direction_reversed": False,
    }

    write_dav2_outputs(tmp_path, relative, disparity, metadata)

    np.testing.assert_array_equal(
        np.load(tmp_path / "photo_dav2_large_relative.npy"), relative
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "photo_dav2_large_disparity.npy"), disparity
    )
    written = json.loads((tmp_path / "dav2_large.json").read_text(encoding="utf-8"))
    assert written == metadata
