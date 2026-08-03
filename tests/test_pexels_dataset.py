from __future__ import annotations

import copy

import pytest

from depthsync.pexels_dataset import select_720p_mp4, validate_manifest


def _video(video_id: int, *, files: list[dict] | None = None) -> dict:
    return {
        "id": video_id,
        "duration": 8,
        "url": f"https://www.pexels.com/video/{video_id}/",
        "user": {"name": "Test Author"},
        "video_files": files
        or [
            {
                "file_type": "video/mp4",
                "width": 1920,
                "height": 1080,
                "link": "https://cdn.example/1080.mp4",
            },
            {
                "file_type": "video/mp4",
                "width": 1280,
                "height": 720,
                "link": "https://cdn.example/720.mp4",
            },
        ],
    }


def _row(scene_id: str, pexels_id: int) -> dict:
    return {
        "scene_id": scene_id,
        "source_type": "pexels",
        "pexels_id": pexels_id,
        "page_url": f"https://www.pexels.com/video/{pexels_id}/",
        "photographer": "Test Author",
        "query": "person walking city",
        "tags": ["walking_turning"],
        "selected_file": {"width": 1280, "height": 720, "file_type": "video/mp4"},
        "duration_seconds": 8.0,
        "sha256": "a" * 64,
        "license_url": "https://www.pexels.com/license/",
    }


def test_select_720p_mp4_prefers_exact_short_side() -> None:
    chosen = select_720p_mp4(_video(1))

    assert chosen["width"] == 1280
    assert chosen["height"] == 720


def test_select_720p_mp4_uses_smallest_larger_file_when_exact_missing() -> None:
    video = _video(
        1,
        files=[
            {"file_type": "video/mp4", "width": 1920, "height": 1080, "link": "a"},
            {"file_type": "video/mp4", "width": 2560, "height": 1440, "link": "b"},
        ],
    )

    chosen = select_720p_mp4(video)

    assert (chosen["width"], chosen["height"]) == (1920, 1080)


def test_manifest_rejects_duplicate_pexels_id_and_secret_text(monkeypatch) -> None:
    manifest = {"schema_version": 1, "scenes": [_row("04", 11), _row("05", 11)]}
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest(manifest)

    manifest = {"schema_version": 1, "scenes": [_row("04", 11)]}
    manifest["scenes"][0]["query"] = "PEXELS_API_KEY=not-allowed"
    with pytest.raises(ValueError, match="secret"):
        validate_manifest(manifest)

    monkeypatch.setenv("PEXELS_API_KEY", "actual-secret")
    manifest = copy.deepcopy(manifest)
    manifest["scenes"][0]["query"] = "actual-secret"
    with pytest.raises(ValueError, match="secret"):
        validate_manifest(manifest)


def test_manifest_accepts_existing_local_scene_without_pexels_fields() -> None:
    manifest = {
        "schema_version": 1,
        "scenes": [
            {
                "scene_id": "01",
                "source_type": "local_existing",
                "tags": ["portrait_close"],
            }
        ],
    }

    validate_manifest(manifest)
