"""Schema and file-selection helpers for the offline Pexels validation set."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


VALID_TAGS = {
    "walking_turning",
    "dance_motion",
    "sport_running",
    "portrait_close",
    "camera_occlusion",
}


def select_720p_mp4(video: dict[str, Any]) -> dict[str, Any]:
    """Choose an exact-720 MP4, otherwise the smallest acceptable MP4."""
    if float(video.get("duration", 0.0)) < 3.0:
        raise ValueError("Pexels video duration must be at least three seconds")
    files = [
        file
        for file in video.get("video_files", [])
        if file.get("file_type") == "video/mp4"
        and isinstance(file.get("width"), int)
        and isinstance(file.get("height"), int)
        and min(file["width"], file["height"]) >= 720
        and file.get("link")
    ]
    if not files:
        raise ValueError("Pexels video has no MP4 with short side at least 720")
    exact = [file for file in files if min(file["width"], file["height"]) == 720]
    candidates = exact or files
    return min(candidates, key=lambda file: (min(file["width"], file["height"]), file["width"] * file["height"]))


def _contains_secret(value: object) -> bool:
    serialized = json.dumps(value, ensure_ascii=False)
    api_key = os.environ.get("PEXELS_API_KEY")
    return "PEXELS_API_KEY" in serialized or bool(api_key and api_key in serialized)


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Reject incomplete, duplicated or secret-bearing manifest data."""
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("manifest schema_version must be 1")
    if _contains_secret(manifest):
        raise ValueError("manifest must not contain a secret")
    rows = manifest.get("scenes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest scenes must be a non-empty list")
    scene_ids: set[str] = set()
    pexels_ids: set[int] = set()
    for row in rows:
        scene_id = row.get("scene_id")
        if not isinstance(scene_id, str) or not re.fullmatch(r"\d{2}", scene_id):
            raise ValueError("manifest scene_id must use two digits")
        if scene_id in scene_ids:
            raise ValueError(f"duplicate scene_id: {scene_id}")
        scene_ids.add(scene_id)
        tags = row.get("tags")
        if not isinstance(tags, list) or not tags or not set(tags).issubset(VALID_TAGS):
            raise ValueError(f"invalid tags for scene {scene_id}")
        source_type = row.get("source_type")
        if source_type == "local_existing":
            continue
        if source_type != "pexels":
            raise ValueError(f"invalid source_type for scene {scene_id}")
        pexels_id = row.get("pexels_id")
        if not isinstance(pexels_id, int):
            raise ValueError(f"missing pexels_id for scene {scene_id}")
        if pexels_id in pexels_ids:
            raise ValueError(f"duplicate Pexels ID: {pexels_id}")
        pexels_ids.add(pexels_id)
        if not str(row.get("page_url", "")).startswith("https://www.pexels.com/"):
            raise ValueError(f"invalid Pexels page URL for scene {scene_id}")
        selected = row.get("selected_file")
        if not isinstance(selected, dict) or selected.get("file_type") != "video/mp4":
            raise ValueError(f"invalid selected MP4 for scene {scene_id}")
        if min(int(selected.get("width", 0)), int(selected.get("height", 0))) < 720:
            raise ValueError(f"selected MP4 is below 720p for scene {scene_id}")
        digest = row.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"invalid SHA-256 for scene {scene_id}")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
