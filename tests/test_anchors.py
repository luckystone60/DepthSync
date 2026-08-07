from __future__ import annotations

import numpy as np
import pytest

from depthsync.anchors import load_anchor_disparity, selected_anchor_path


def test_dav2_anchor_path_is_default_contract(tmp_path) -> None:
    assert selected_anchor_path(tmp_path, "dav2-large").name == "photo_dav2_large_disparity.npy"
    assert selected_anchor_path(tmp_path, "depthpro").name == "photo_disparity.npy"


def test_load_anchor_rejects_missing_or_non_finite_values(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing"):
        load_anchor_disparity(tmp_path, "dav2-large")

    np.save(tmp_path / "photo_dav2_large_disparity.npy", np.asarray([[np.nan]], np.float32))
    with pytest.raises(ValueError, match="finite"):
        load_anchor_disparity(tmp_path, "dav2-large")
