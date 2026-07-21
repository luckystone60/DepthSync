import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from depthsync import DepthSync, DepthSyncConfig, FrameParameters, MotionSequence
from depthsync.validation import prepare_validation_clip
from depthsync.depth_visualization import colorize_depth, depth_to_gray16, normalize_depth


class DepthSyncTest(unittest.TestCase):
    def test_anchor_scale_and_temporal_jump_are_improved(self):
        h, w, t, anchor = 72, 96, 7, 3
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        truth = 1.5 + 0.004 * xx + 0.002 * yy + 0.25 * (np.hypot(xx - 48, yy - 36) < 18)
        photo = truth + 0.025 * np.sin(xx * 0.8) * np.sin(yy * 0.7)
        low = cv2.resize(truth, (48, 36), interpolation=cv2.INTER_AREA)
        video = np.stack([(low - 0.12 * i) / (0.72 + 0.07 * i) for i in range(t)])
        sync = DepthSync(DepthSyncConfig(depth_mode="depth", min_fit_pixels=64, sample_count=512))
        result = sync(video, photo, anchor)
        photo_low = cv2.resize(photo, (48, 36), interpolation=cv2.INTER_AREA)
        before_error = np.mean(np.abs(video[anchor] - photo_low))
        after_error = np.mean(np.abs(result.depths[anchor] - photo_low))
        self.assertLess(after_error, before_error * 0.35)
        before_jump = np.mean(np.abs(np.diff(video, axis=0)))
        after_jump = np.mean(np.abs(np.diff(result.depths, axis=0)))
        self.assertLess(after_jump, before_jump)
        self.assertEqual(result.depths.shape, (t, 36, 48))
        high_res = sync.apply_frame(video[anchor], result.parameters[anchor], (h, w))
        self.assertEqual(high_res.shape, (h, w))

    def test_normalized_block_motion_is_accepted(self):
        t, h, w = 5, 24, 32
        base = np.tile(np.linspace(0.2, 1.0, w, dtype=np.float32), (h, 1))
        video = np.stack([np.roll(base, i, axis=1) for i in range(t)])
        fields_prev = np.zeros((t, 4, 6, 2), np.float32)
        fields_next = np.zeros_like(fields_prev)
        fields_prev[1:, ..., 0] = -1.0 / (w - 1)
        fields_next[:-1, ..., 0] = 1.0 / (w - 1)
        motion = MotionSequence(fields_prev, fields_next)
        result = DepthSync(DepthSyncConfig(min_fit_pixels=32, sample_count=256)).offline_prepare(video, base, 2, motion)
        self.assertTrue(np.all(np.isfinite(result.scales)))
        self.assertEqual(len(result.fallback_reasons), t)
        self.assertEqual(result.scales[1], result.scales[2])
        self.assertEqual(result.scales[3], result.scales[2])
        self.assertEqual(result.fallback_reasons[1], "anchor_lock")
        self.assertEqual(result.fallback_reasons[3], "anchor_lock")

    def test_v3_improves_nonlinear_and_spatial_anchor_alignment(self):
        t, h, w, anchor = 7, 54, 80, 3
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        raw_anchor = 0.15 + 0.75 * xx / (w - 1) + 0.1 * yy / (h - 1)
        video = np.stack([raw_anchor + 0.005 * (i - anchor) for i in range(t)])
        spatial_bias = 0.10 * np.sin(2.0 * np.pi * xx / w) * np.cos(np.pi * yy / h)
        photo = 0.25 + 0.7 * raw_anchor + 0.55 * raw_anchor**2 + spatial_bias
        common = dict(depth_mode="disparity", min_fit_pixels=64, sample_count=1024)
        affine = DepthSync(
            DepthSyncConfig(
                **common,
                mapping_mode="affine",
                residual_grid_shape=(0, 0),
                residual_radius=0,
            )
        ).offline_prepare(video, photo, anchor)
        v3_sync = DepthSync(
            DepthSyncConfig(
                **common,
                mapping_mode="lut",
                lut_nodes=8,
                residual_grid_shape=(9, 16),
                residual_radius=3,
            )
        )
        v3 = v3_sync.offline_prepare(video, photo, anchor)
        affine_error = float(np.median(np.abs(affine.depths[anchor] - photo)))
        v3_error = float(np.median(np.abs(v3.depths[anchor] - photo)))
        self.assertLess(v3_error, affine_error * 0.45)
        self.assertTrue(np.all(np.diff(v3.lut_y, axis=1) >= -1e-6))
        self.assertGreater(float(np.mean(np.abs(v3.residual_grids[anchor]))), 0.0)
        self.assertGreater(float(np.mean(np.abs(v3.residual_grids[anchor - 1]))), 0.0)
        self.assertEqual(float(np.max(np.abs(v3.residual_grids[anchor - 3]))), 0.0)
        replay = v3_sync.apply_frame(video[anchor], v3.parameters[anchor])
        np.testing.assert_allclose(replay, v3.depths[anchor], atol=1e-6)

    def test_apply_frame_uses_only_scale_and_offset(self):
        depth = np.array([[1.0, 2.0]], np.float32)
        result = DepthSync().apply_frame(depth, FrameParameters(2.0, -0.5, 1.0))
        np.testing.assert_allclose(result, [[1.5, 3.5]])

    def test_zero_is_valid_for_relative_disparity(self):
        depth = np.array([[0.0, 1.0]], np.float32)
        result = DepthSync().apply_frame(depth, FrameParameters(2.0, 0.25, 1.0))
        np.testing.assert_allclose(result, [[0.25, 2.25]])

    def test_depth_visualization_uses_explicit_limits(self):
        depth = np.array([[0.0, 0.5, 1.0]], np.float32)
        normalized = normalize_depth(depth, (0.0, 1.0))
        np.testing.assert_allclose(normalized, depth)
        self.assertEqual(colorize_depth(depth, (0.0, 1.0)).shape, (1, 3, 3))
        gray16 = depth_to_gray16(depth, (0.0, 1.0))
        self.assertEqual(gray16.dtype, np.uint16)
        self.assertEqual(int(gray16[0, -1]), 65535)

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
            self.assertAlmostEqual(manifest.sample_timestamps_seconds[45] - manifest.clip_start_seconds, 1.5)


if __name__ == "__main__":
    unittest.main()
