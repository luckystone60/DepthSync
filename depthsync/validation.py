"""Deterministic validation data preparation for the three local videos."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2

from .pexels_dataset import validate_manifest


@dataclass(frozen=True)
class ClipManifest:
    source: str
    source_sha256: str
    source_fps: float
    source_frame_count: int
    source_width: int
    source_height: int
    source_duration_seconds: float
    clip_start_seconds: float
    clip_duration_seconds: float
    target_fps: float
    target_frame_count: int
    anchor_index: int
    sample_timestamps_seconds: list[float]
    clip_width: int
    clip_height: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scaled_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, max_side / max(width, height))
    out_width = max(2, int(round(width * scale / 2.0)) * 2)
    out_height = max(2, int(round(height * scale / 2.0)) * 2)
    return out_width, out_height


def prepare_validation_clip(
    source: Path,
    output_dir: Path,
    duration_seconds: float = 3.0,
    target_fps: float = 30.0,
    max_side: int = 1280,
) -> ClipManifest:
    """Create a centered 3 s / 90 frame clip and its full-resolution anchor."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Invalid video metadata: {source}")
    source_duration = frame_count / source_fps
    if source_duration + 1e-6 < duration_seconds:
        capture.release()
        raise ValueError(f"Video is shorter than {duration_seconds} seconds: {source}")

    target_count = int(round(duration_seconds * target_fps))
    anchor_index = target_count // 2
    start = max(0.0, (source_duration - duration_seconds) * 0.5)
    timestamps = [start + i / target_fps for i in range(target_count)]
    out_width, out_height = _scaled_size(width, height, max_side)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_path = output_dir / "clip.mp4"
    anchor_path = output_dir / "anchor.png"
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (out_width, out_height))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Cannot create validation clip: {clip_path}")

    try:
        for index, timestamp in enumerate(timestamps):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode {source} at {timestamp:.6f} seconds")
            if index == anchor_index and not cv2.imwrite(str(anchor_path), frame):
                raise ValueError(f"Cannot write anchor frame: {anchor_path}")
            if (width, height) != (out_width, out_height):
                frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
        capture.release()

    manifest = ClipManifest(
        source=str(source.resolve()),
        source_sha256=_sha256(source),
        source_fps=source_fps,
        source_frame_count=frame_count,
        source_width=width,
        source_height=height,
        source_duration_seconds=source_duration,
        clip_start_seconds=start,
        clip_duration_seconds=duration_seconds,
        target_fps=target_fps,
        target_frame_count=target_count,
        anchor_index=anchor_index,
        sample_timestamps_seconds=timestamps,
        clip_width=out_width,
        clip_height=out_height,
    )
    (output_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def prepare_all(sources: Iterable[Path], output_root: Path) -> list[ClipManifest]:
    return [prepare_validation_clip(source, output_root / source.stem) for source in sources]


def scene_ids_from_manifest(path: Path | str, expected_count: int = 20) -> list[str]:
    """Read a validation manifest and require contiguous two-digit scene IDs."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(payload)
    actual = sorted(row["scene_id"] for row in payload["scenes"])
    expected = [f"{index:02d}" for index in range(1, expected_count + 1)]
    missing = [scene for scene in expected if scene not in actual]
    unexpected = [scene for scene in actual if scene not in expected]
    if missing or unexpected or len(actual) != expected_count:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        raise ValueError("manifest scene IDs must be contiguous: " + "; ".join(details))
    return expected


def _validate_box(name: str, value: Any, scene: str) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} missing for scene {scene}")
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} invalid for scene {scene}") from error
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"{name} invalid for scene {scene}")


def validate_scene_coverage(config: dict[str, Any], scene_ids: Iterable[str]) -> None:
    """Require manually supplied, usable ROIs and inspection frames per scene."""
    for scene in scene_ids:
        row = config.get(scene)
        if not isinstance(row, dict):
            raise ValueError(f"validation config missing scene {scene}")
        _validate_box("face_box", row.get("face_box"), scene)
        _validate_box("subject_box", row.get("subject_box"), scene)
        inspection = row.get("inspection_frames")
        if not isinstance(inspection, list) or not {0, 45}.issubset(inspection):
            raise ValueError(f"inspection_frames missing for scene {scene}")
        tail = row.get("tail_frames")
        if not isinstance(tail, list) or tail != list(range(75, 90)):
            raise ValueError(f"tail_frames missing for scene {scene}")


def prepare_from_manifest(
    manifest_path: Path,
    source_root: Path,
    output_root: Path,
    scene_config: Path | None = None,
    expected_count: int = 20,
) -> list[ClipManifest]:
    scenes = scene_ids_from_manifest(manifest_path, expected_count)
    if scene_config is not None:
        validate_scene_coverage(
            json.loads(scene_config.read_text(encoding="utf-8")), scenes
        )
    sources = [source_root / f"{scene}.mp4" for scene in scenes]
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError("missing source videos: " + ", ".join(missing))
    return prepare_all(sources, output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic 3 s / 90 frame DepthSync validation clips")
    parser.add_argument("sources", nargs="*", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/clips"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("testdata"))
    parser.add_argument("--scene-config", type=Path, default=Path("config/validation-scenes.json"))
    parser.add_argument("--expected-count", type=int, default=20)
    args = parser.parse_args()
    if args.manifest is not None:
        if args.sources:
            parser.error("sources cannot be combined with --manifest")
        manifests = prepare_from_manifest(
            args.manifest,
            args.source_root,
            args.output_root,
            args.scene_config,
            args.expected_count,
        )
    elif args.sources:
        manifests = prepare_all(args.sources, args.output_root)
    else:
        parser.error("provide sources or --manifest")
    for manifest in manifests:
        print(f"{Path(manifest.source).name}: {manifest.target_frame_count} frames, anchor={manifest.anchor_index}")


if __name__ == "__main__":
    main()
