from __future__ import annotations

import json

import numpy as np

from tools.run_dav2_validation import dav2_output_complete


def test_dav2_output_complete_requires_all_contract_files(tmp_path) -> None:
    assert dav2_output_complete(tmp_path) is False
    np.save(tmp_path / "photo_dav2_large_relative.npy", np.ones((2, 2), np.float32))
    np.save(tmp_path / "photo_dav2_large_disparity.npy", np.ones((2, 2), np.float32))
    (tmp_path / "dav2_large.json").write_text(json.dumps({"backend": "Depth-Anything-V2-Large"}), encoding="utf-8")

    assert dav2_output_complete(tmp_path) is True
