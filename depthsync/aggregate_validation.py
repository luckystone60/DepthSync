"""Aggregate per-scene DAv2 validation metrics without hiding failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .pexels_dataset import validate_manifest


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "median": float("nan"), "p90": float("nan")}
    return {
        "count": len(values),
        "median": round(float(np.median(values)), 8),
        "p90": round(float(np.quantile(values, 0.9)), 8),
    }


def aggregate_validation(result_root: Path | str, manifest_path: Path | str) -> dict[str, Any]:
    root = Path(result_root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    complete: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for row in manifest["scenes"]:
        scene = row["scene_id"]
        metrics_path = root / scene / "metrics.json"
        if not metrics_path.is_file():
            incomplete.append(scene)
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            raw = float(metrics["raw_anchor_nmae"])
            local = float(metrics["local_anchor_nmae"])
            item = {
                "scene": scene,
                "tags": row["tags"],
                "anchor_improvement": raw - local,
                "temporal_regression": float(metrics["local_temporal_p95"]) - float(metrics["raw_temporal_p95"]),
                "local_stage_temporal_regression": float(metrics["local_temporal_p95"]) - float(metrics["v4_temporal_p95"]),
                "local_strength": float(metrics["local_strength"]),
                "fallback_frames": int(metrics.get("v4_fallback_frames", 0)),
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            incomplete.append(scene)
            continue
        complete.append(item)
    tags: dict[str, dict[str, float | int]] = {}
    for tag in sorted({tag for item in complete for tag in item["tags"]}):
        items = [item for item in complete if tag in item["tags"]]
        anchor = _summary([item["anchor_improvement"] for item in items])
        temporal = _summary([item["temporal_regression"] for item in items])
        local_temporal = _summary([item["local_stage_temporal_regression"] for item in items])
        tags[tag] = {
            "count": len(items),
            "anchor_improvement_median": anchor["median"],
            "anchor_improvement_p90": anchor["p90"],
            "temporal_regression_median": temporal["median"],
            "temporal_regression_p90": temporal["p90"],
            "local_stage_temporal_regression_median": local_temporal["median"],
            "local_stage_temporal_regression_p90": local_temporal["p90"],
            "local_strength_median": round(float(np.median([item["local_strength"] for item in items])), 8),
        }
    return {
        "scene_count": len(manifest["scenes"]),
        "completed_count": len(complete),
        "incomplete": incomplete,
        "tags": tags,
        "worst_anchor_improvement": sorted(complete, key=lambda item: item["anchor_improvement"])[:3],
        "worst_temporal_regression": sorted(complete, key=lambda item: item["temporal_regression"], reverse=True)[:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate_validation(args.result_root, args.manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
