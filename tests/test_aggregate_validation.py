from __future__ import annotations

import json

from depthsync.aggregate_validation import aggregate_validation


def test_aggregate_reports_per_tag_and_incomplete_scenes(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenes": [
                    {"scene_id": "01", "source_type": "local_existing", "tags": ["dance_motion"]},
                    {"scene_id": "02", "source_type": "local_existing", "tags": ["dance_motion"]},
                    {"scene_id": "03", "source_type": "local_existing", "tags": ["portrait_close"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "results"
    for scene, gain in (("01", 0.03), ("02", 0.01)):
        directory = root / scene
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(
            json.dumps(
                {
                    "v4_anchor_nmae": 0.10,
                    "v5_anchor_nmae": 0.10 - gain,
                    "v4_subject_anchor_nmae": 0.12,
                    "v5_subject_anchor_nmae": 0.10,
                    "v4_temporal_p95": 0.02,
                    "v5_temporal_p95": 0.021,
                    "v4_fallback_frames": 0,
                }
            ),
            encoding="utf-8",
        )

    report = aggregate_validation(root, manifest)

    assert report["incomplete"] == ["03"]
    assert report["tags"]["dance_motion"]["count"] == 2
    assert report["tags"]["dance_motion"]["anchor_improvement_median"] == 0.02
    assert report["worst_anchor_improvement"][0]["scene"] == "02"
