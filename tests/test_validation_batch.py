from __future__ import annotations

import json

import pytest

from depthsync.validation import scene_ids_from_manifest, validate_scene_coverage


def test_scene_ids_require_contiguous_01_through_20(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenes": [
                    {"scene_id": "01", "source_type": "local_existing", "tags": ["portrait_close"]},
                    {"scene_id": "03", "source_type": "local_existing", "tags": ["dance_motion"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="02"):
        scene_ids_from_manifest(path, expected_count=3)


def test_validation_scene_requires_subject_and_inspection_fields() -> None:
    with pytest.raises(ValueError, match="subject_box"):
        validate_scene_coverage(
            {"04": {"face_box": [0.0, 0.0, 1.0, 1.0]}},
            ["04"],
        )


def test_validation_scene_accepts_manual_subject_and_tail_frames() -> None:
    validate_scene_coverage(
        {
            "04": {
                "face_box": [0.2, 0.2, 0.4, 0.5],
                "subject_box": [0.1, 0.1, 0.8, 0.9],
                "inspection_frames": [0, 45, 60],
                "tail_frames": list(range(75, 90)),
            }
        },
        ["04"],
    )
