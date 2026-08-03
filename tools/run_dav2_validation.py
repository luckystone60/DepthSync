"""Batch DAv2-Large anchor inference for a validated DepthSync corpus."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from depthsync.validation import scene_ids_from_manifest


def dav2_output_complete(output_dir: Path | str) -> bool:
    directory = Path(output_dir)
    relative_path = directory / "photo_dav2_large_relative.npy"
    disparity_path = directory / "photo_dav2_large_disparity.npy"
    metadata_path = directory / "dav2_large.json"
    if not all(path.is_file() for path in (relative_path, disparity_path, metadata_path)):
        return False
    try:
        relative = np.load(relative_path)
        disparity = np.load(disparity_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        relative.ndim == 2
        and disparity.shape == relative.shape
        and np.all(np.isfinite(relative))
        and np.all(np.isfinite(disparity))
        and metadata.get("backend") == "Depth-Anything-V2-Large"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "config" / "pexels-validation-manifest.json")
    parser.add_argument("--clip-root", type=Path, default=ROOT / "artifacts" / "clips")
    parser.add_argument("--depth-root", type=Path, default=ROOT / "artifacts" / "depth")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--expected-count", type=int, default=20)
    parser.add_argument("--model-id", default="depth-anything/Depth-Anything-V2-Large-hf")
    parser.add_argument("--revision", default="7581137eff8d4e94f6e796d3baea0e9fa79b22d2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    scenes = scene_ids_from_manifest(args.manifest, args.expected_count)
    outcomes: dict[str, dict[str, str]] = {}
    for scene in scenes:
        output_dir = args.depth_root / scene
        if not args.force and dav2_output_complete(output_dir):
            outcomes[scene] = {"status": "cached"}
            continue
        command = [
            str(args.python),
            str(ROOT / "tools" / "run_depth_models.py"),
            "dav2-large",
            "--anchor",
            str(args.clip_root / scene / "anchor.png"),
            "--reference-vda",
            str(output_dir / "video_disparity.npz"),
            "--output-dir",
            str(output_dir),
            "--model-id",
            args.model_id,
            "--revision",
            args.revision,
            "--device",
            args.device,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            outcomes[scene] = {"status": "complete"}
        except subprocess.CalledProcessError as error:
            outcomes[scene] = {"status": "failed", "error": error.stderr[-1000:]}
    args.depth_root.mkdir(parents=True, exist_ok=True)
    (args.depth_root / "dav2-batch.json").write_text(
        json.dumps(outcomes, indent=2), encoding="utf-8"
    )
    failed = [scene for scene, outcome in outcomes.items() if outcome["status"] == "failed"]
    if failed:
        raise SystemExit("DAv2 failed for: " + ", ".join(failed))


if __name__ == "__main__":
    main()
