"""Cache SEA-RAFT-S bidirectional flow for validation clips."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import time
from pathlib import Path

import cv2

from depthsync.flow import save_dense_flow
from depthsync.sea_raft import SeaRaftEstimator


def _peak_working_set_bytes() -> int | None:
    if not hasattr(ctypes, "windll"):
        return None
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return int(counters.PeakWorkingSetSize) if ok else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_frames(path: Path, max_pairs: int | None) -> list:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    limit = None if max_pairs is None else max_pairs + 1
    frames = []
    while limit is None or len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < 2:
        raise ValueError(f"Video has fewer than two readable frames: {path}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-root", type=Path, default=Path("artifacts/clips"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/flow"))
    parser.add_argument("--repository", type=Path, default=Path("third_party/SEA-RAFT"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sea_raft_s.safetensors"))
    parser.add_argument("--scenes", nargs="+", default=["01", "02", "03"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-pairs", type=int)
    args = parser.parse_args()

    estimator = SeaRaftEstimator(args.checkpoint, args.repository, device=args.device)
    checkpoint_hash = _sha256(args.checkpoint)
    for scene in args.scenes:
        frames = _read_frames(args.clip_root / scene / "clip.mp4", args.max_pairs)
        peak_gpu_bytes = None
        if args.device.startswith("cuda"):
            import torch
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        flow = estimator.estimate_pairs(frames)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if args.device.startswith("cuda"):
            peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
        output_dir = args.output_root / scene
        output_dir.mkdir(parents=True, exist_ok=True)
        save_dense_flow(output_dir / "sea_raft_s_flow.npz", flow)
        timing = {
            "scene": scene, "frame_count": len(frames),
            "pair_count_each_direction": len(frames) - 1,
            "bidirectional_inference_count": 2 * (len(frames) - 1),
            "elapsed_ms": elapsed_ms,
            "ms_per_bidirectional_pair": elapsed_ms / (len(frames) - 1),
            "device": args.device, "input_shape": [144, 256],
            "source_revision": "9137517ba24e628442aec097d3afe71d03503b75",
            "checkpoint_sha256": checkpoint_hash,
            "peak_working_set_bytes": _peak_working_set_bytes(),
            "peak_gpu_allocated_bytes": peak_gpu_bytes,
        }
        (output_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        print(f"{scene}: {len(frames)-1} pairs -> {output_dir} ({elapsed_ms:.1f} ms)")


if __name__ == "__main__":
    main()
