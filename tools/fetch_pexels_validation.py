"""Search and reproducibly download Pexels videos for offline validation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from depthsync.pexels_dataset import VALID_TAGS, select_720p_mp4, sha256_file, validate_manifest


QUERIES: dict[str, tuple[str, ...]] = {
    "walking_turning": ("full body person walking city", "person turning street"),
    "dance_motion": ("full body dancer moving", "dance complex background"),
    "sport_running": ("person running action", "athlete moving outdoors"),
    "portrait_close": ("portrait person hair hands", "person close up movement"),
    "camera_occlusion": ("person camera movement foreground", "person walking foreground occlusion"),
}


def _api_get(endpoint: str, params: dict[str, object]) -> dict[str, Any]:
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        raise RuntimeError("PEXELS_API_KEY is required")
    request = Request(
        f"https://api.pexels.com/v1/videos/{endpoint}?{urlencode(params)}",
        headers={"Authorization": api_key, "User-Agent": "DepthSync-offline-validation/1.0"},
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def list_candidates(tag: str, per_query: int) -> list[dict[str, Any]]:
    if tag not in VALID_TAGS:
        raise ValueError(f"unknown tag: {tag}")
    candidates: dict[int, dict[str, Any]] = {}
    for query in QUERIES[tag]:
        payload = _api_get("search", {"query": query, "per_page": per_query})
        for video in payload.get("videos", []):
            try:
                selected = select_720p_mp4(video)
            except ValueError:
                continue
            candidates[int(video["id"])] = {
                "pexels_id": int(video["id"]),
                "page_url": video["url"],
                "photographer": video.get("user", {}).get("name", ""),
                "duration_seconds": float(video["duration"]),
                "selected_file": {
                    "file_type": selected["file_type"],
                    "width": selected["width"],
                    "height": selected["height"],
                    "link": selected["link"],
                },
                "query": query,
                "tags": [tag],
            }
    return list(candidates.values())


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def _download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "DepthSync-offline-validation/1.0"})
    with urlopen(request, timeout=120) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def accept_candidate(
    scene_id: str,
    tag: str,
    pexels_id: int,
    manifest_path: Path,
    output_root: Path,
    replace: bool,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    if scene_id in {row["scene_id"] for row in manifest["scenes"]}:
        raise ValueError(f"scene already exists in manifest: {scene_id}")
    if pexels_id in {row.get("pexels_id") for row in manifest["scenes"]}:
        raise ValueError(f"Pexels ID already exists in manifest: {pexels_id}")
    video = _api_get(f"videos/{pexels_id}", {})
    selected = select_720p_mp4(video)
    destination = output_root / f"{scene_id}.mp4"
    if destination.exists() and not replace:
        raise FileExistsError(f"source already exists: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".mp4", prefix=f".{scene_id}-", dir=output_root, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        _download(selected["link"], temporary)
        capture = cv2.VideoCapture(str(temporary))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if fps <= 0.0 or frames / fps < 3.0:
            raise ValueError("downloaded Pexels video is shorter than three seconds")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    row = {
        "scene_id": scene_id,
        "source_type": "pexels",
        "pexels_id": pexels_id,
        "page_url": video["url"],
        "photographer": video.get("user", {}).get("name", ""),
        "query": QUERIES[tag][0],
        "tags": [tag],
        "selected_file": {
            "file_type": selected["file_type"],
            "width": selected["width"],
            "height": selected["height"],
        },
        "duration_seconds": float(video["duration"]),
        "sha256": sha256_file(destination),
        "license_url": "https://www.pexels.com/license/",
    }
    manifest["scenes"].append(row)
    validate_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "config" / "pexels-validation-manifest.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "testdata")
    parser.add_argument("--list-candidates", action="store_true")
    parser.add_argument("--tag", choices=sorted(VALID_TAGS), required=True)
    parser.add_argument("--per-query", type=int, default=15)
    parser.add_argument("--accept", type=int)
    parser.add_argument("--scene")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.list_candidates:
        print(json.dumps(list_candidates(args.tag, args.per_query), indent=2))
        return
    if args.accept is None or args.scene is None:
        parser.error("use --list-candidates or both --accept and --scene")
    row = accept_candidate(args.scene, args.tag, args.accept, args.manifest, args.output_root, args.replace)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
