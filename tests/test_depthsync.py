import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from depthsync import DepthSync, DepthSyncConfig, MotionSequence
from depthsync.core import _adjacent_distribution_jump
from depthsync.depth_visualization import colorize_depth, depth_to_gray16, normalize_depth
from depthsync.validation import prepare_validation_clip


class DepthSyncTest(unittest.TestCase):
    def test_distribution_jump_ignores_scale_but_detects_occlusion(self):
        cfg = DepthSyncConfig(min_fit_pixels=32)
        base = np.tile(np.linspace(0.2, 8.0, 80, dtype=np.float32), (48, 1))
        scaled = base * 2.5 + 7.0
        occluded = base.copy()
        occluded[:, :52] = 30.0

        self.assertLess(_adjacent_distribution_jump(scaled, base, cfg), 0.01)
        self.assertGreater(_adjacent_distribution_jump(occluded, base, cfg), 0.05)

    def test_v41_anchor_alignment_and_replay(self):
        h, w, t, anchor = 48, 80, 7, 3
        source = np.tile(np.linspace(0.2, 8.0, w, dtype=np.float32), (h, 1))
        video = np.stack([source * (0.98 + 0.01 * index) for index in range(t)])
        photo = 5.0 + 3.0 * source + 0.6 * source**2
        sync = DepthSync(DepthSyncConfig(min_fit_pixels=64, sample_count=4096))

        result = sync.offline_prepare(video, photo, anchor)

        raw_error = float(np.median(np.abs(video[anchor] - photo)))
        mapped_error = float(np.median(np.abs(result.depths[anchor] - photo)))
        self.assertLess(mapped_error, raw_error * 0.1)
        self.assertTrue(np.all(np.diff(result.lut_y, axis=1) >= -1e-6))
        replay = sync.apply_frame(video[anchor], result.parameters[anchor])
        np.testing.assert_allclose(replay, result.depths[anchor], atol=1e-6)

    def test_normalized_block_motion_is_accepted(self):
        t, h, w = 5, 24, 32
        base = np.tile(np.linspace(0.2, 1.0, w, dtype=np.float32), (h, 1))
        video = np.stack([np.roll(base, i, axis=1) for i in range(t)])
        fields_prev = np.zeros((t, 4, 6, 2), np.float32)
        fields_next = np.zeros_like(fields_prev)
        fields_prev[1:, ..., 0] = -1.0 / (w - 1)
        fields_next[:-1, ..., 0] = 1.0 / (w - 1)
        motion = MotionSequence(fields_prev, fields_next)

        result = DepthSync(
            DepthSyncConfig(min_fit_pixels=32, sample_count=256)
        ).offline_prepare(video, base, 2, motion)

        self.assertTrue(np.all(np.isfinite(result.scales)))
        self.assertEqual(result.lut_x.shape, (t, 64))

    def test_v41_adaptive_lut_is_no_worse_than_fixed_candidates(self):
        h, w = 48, 80
        source = np.tile(np.linspace(0.2, 4.0, w, dtype=np.float32), (h, 1))
        photo = 2.0 + 1.8 * source + 0.4 * source**2
        video = np.stack([source, source, source])
        errors = {}
        for candidates in ((8,), (16,), (8, 16)):
            result = DepthSync(
                DepthSyncConfig(
                    lut_nodes=16,
                    lut_candidate_nodes=candidates,
                    subject_offset_clip_fraction=0.0,
                    min_fit_pixels=64,
                    sample_count=1024,
                )
            ).offline_prepare(video, photo, 1)
            errors[candidates] = float(np.median(np.abs(result.depths[1] - photo)))
        self.assertLessEqual(errors[(8, 16)], min(errors[(8,)], errors[(16,)]) + 1e-6)

    def test_v41_recovers_large_nonlinear_range_with_invalid_floor(self):
        h, w = 72, 128
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        source = 18.0 * (0.75 * xx / (w - 1) + 0.25 * yy / (h - 1))
        source[:26] = 0.0
        photo = 8.0 + 4.0 * source + 1.15 * source**2
        photo[:26] = 0.0
        video = np.stack([source - 0.02, source, source + 0.02])
        result = DepthSync(
            DepthSyncConfig(subject_offset_clip_fraction=0.0)
        ).offline_prepare(video, photo, 1)
        valid = source > 0.0
        raw_error = float(np.median(np.abs(source[valid] - photo[valid])))
        mapped_error = float(np.median(np.abs(result.depths[1][valid] - photo[valid])))
        self.assertLess(mapped_error, raw_error * 0.08)
        self.assertGreater(float(result.lut_y[1, -1]), 200.0)

    def test_v41_quantile_fallback_aligns_unpaired_distributions(self):
        h, w = 64, 96
        source = np.tile(np.linspace(0.2, 10.0, w, dtype=np.float32), (h, 1))
        target_distribution = 20.0 + 3.0 * source + 0.8 * source**2
        photo = np.flip(target_distribution, axis=1).copy()
        video = np.stack([source, source, source])
        result = DepthSync(
            DepthSyncConfig(subject_offset_clip_fraction=0.0)
        ).offline_prepare(video, photo, 1)
        before = np.quantile(source, [0.1, 0.5, 0.9])
        after = np.quantile(result.depths[1], [0.1, 0.5, 0.9])
        target = np.quantile(photo, [0.1, 0.5, 0.9])
        self.assertLess(
            float(np.mean(np.abs(after - target))),
            float(np.mean(np.abs(before - target))) * 0.1,
        )

    def test_v41_distribution_stabilization_reduces_scale_drift(self):
        h, w, t, anchor = 48, 80, 9, 4
        base = np.tile(np.linspace(0.2, 8.0, w, dtype=np.float32), (h, 1))
        scales = np.linspace(0.7, 1.3, t, dtype=np.float32)
        video = np.stack([base * scale for scale in scales])
        photo = 10.0 + 5.0 * base + base**2
        common = dict(sample_count=4096, subject_offset_clip_fraction=0.0)
        fixed = DepthSync(
            DepthSyncConfig(**common, enable_lut_distribution_stabilization=False)
        ).offline_prepare(video, photo, anchor)
        stabilized = DepthSync(DepthSyncConfig(**common)).offline_prepare(video, photo, anchor)
        self.assertLess(
            float(np.ptp(np.median(stabilized.depths, axis=(1, 2)))),
            float(np.ptp(np.median(fixed.depths, axis=(1, 2)))) * 0.55,
        )

    def test_depth_visualization_uses_explicit_limits(self):
        depth = np.array([[0.0, 0.5, 1.0]], np.float32)
        np.testing.assert_allclose(normalize_depth(depth, (0.0, 1.0)), depth)
        self.assertEqual(colorize_depth(depth, (0.0, 1.0)).shape, (1, 3, 3))
        self.assertEqual(int(depth_to_gray16(depth, (0.0, 1.0))[0, -1]), 65535)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            DepthSync()([], np.ones((4, 4), np.float32), 0)

    def test_validation_clip_has_exact_timeline(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (32, 24))
            for index in range(80):
                writer.write(np.full((24, 32, 3), index, np.uint8))
            writer.release()
            manifest = prepare_validation_clip(source, root / "clip", max_side=32)
            capture = cv2.VideoCapture(str(root / "clip" / "clip.mp4"))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 90)
            self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 30.0)
            capture.release()
            self.assertEqual(manifest.anchor_index, 45)


if __name__ == "__main__":
    unittest.main()
