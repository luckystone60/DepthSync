import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from depthsync import DepthSync, DepthSyncConfig, FrameParameters, MotionSequence
from depthsync.core import _largest_open_component, _static_guidance_mask
from depthsync.validation import prepare_validation_clip
from depthsync.depth_visualization import colorize_depth, depth_to_gray16, normalize_depth


class DepthSyncTest(unittest.TestCase):
    def test_distribution_jump_ignores_scale_but_detects_occlusion(self):
        from depthsync.core import _adjacent_distribution_jump

        cfg = DepthSyncConfig(min_fit_pixels=32)
        base = np.tile(np.linspace(0.2, 8.0, 80, dtype=np.float32), (48, 1))
        scaled = base * 2.5 + 7.0
        occluded = base.copy()
        occluded[:, :52] = 30.0
        self.assertLess(_adjacent_distribution_jump(base, scaled, cfg), 1e-5)
        self.assertGreater(
            _adjacent_distribution_jump(base, occluded, cfg),
            cfg.lut_frame_freeze_jump_threshold,
        )

    def test_anchor_scale_and_temporal_jump_are_improved(self):
        h, w, t, anchor = 72, 96, 7, 3
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        truth = 1.5 + 0.004 * xx + 0.002 * yy + 0.25 * (np.hypot(xx - 48, yy - 36) < 18)
        photo = truth + 0.025 * np.sin(xx * 0.8) * np.sin(yy * 0.7)
        low = cv2.resize(truth, (48, 36), interpolation=cv2.INTER_AREA)
        video = np.stack([(low - 0.12 * i) / (0.72 + 0.07 * i) for i in range(t)])
        sync = DepthSync(DepthSyncConfig(depth_mode="depth", min_fit_pixels=64, sample_count=512))
        result = sync(video, photo, anchor)
        self.assertEqual(result.region_labels.size, 0)
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
        self.assertEqual(result.fallback_reasons[1], "")
        self.assertEqual(result.fallback_reasons[3], "")
        self.assertEqual(result.region_labels.size, 0)
        self.assertEqual(result.lut_x.shape[1], 64)

    def test_v3_improves_nonlinear_and_spatial_anchor_alignment(self):
        t, h, w, anchor = 7, 54, 80, 3
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        raw_anchor = 0.15 + 0.75 * xx / (w - 1) + 0.1 * yy / (h - 1)
        video = np.stack([raw_anchor + 0.005 * (i - anchor) for i in range(t)])
        spatial_bias = 0.10 * np.sin(2.0 * np.pi * xx / w) * np.cos(np.pi * yy / h)
        photo = 0.25 + 0.7 * raw_anchor + 0.55 * raw_anchor**2 + spatial_bias
        common = dict(
            algorithm_version="v3",
            depth_mode="disparity",
            min_fit_pixels=64,
            sample_count=1024,
        )
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
        np.testing.assert_allclose(v3.lut_x, np.repeat(v3.lut_x[anchor][None], t, axis=0), atol=1e-6)
        relative_scale_delta = np.abs(np.diff(v3.scales)) / np.maximum(np.abs(v3.scales[:-1]), 1e-6)
        self.assertLessEqual(float(np.max(relative_scale_delta)), v3_sync.cfg.lut_max_scale_delta + 1e-6)
        self.assertGreater(float(np.mean(np.abs(v3.residual_grids[anchor]))), 0.0)
        self.assertGreater(float(np.mean(np.abs(v3.residual_grids[anchor - 1]))), 0.0)
        self.assertEqual(float(np.max(np.abs(v3.residual_grids[anchor - 3]))), 0.0)
        self.assertEqual(v3.static_mask.shape, (72, 128))
        self.assertEqual(v3.static_target_grid.shape, (72, 128))
        replay = v3_sync.apply_frame(video[anchor], v3.parameters[anchor])
        np.testing.assert_allclose(replay, v3.depths[anchor], atol=1e-6)

    def test_static_photo_guidance_removes_background_drift(self):
        t, h, w, anchor = 9, 36, 48, 4
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        photo = 0.2 + 0.4 * xx / (w - 1) + 0.15 * yy / (h - 1)
        video = np.stack([photo + 0.02 * (i - anchor) for i in range(t)])
        fields = np.zeros((t, 6, 8, 2), np.float32)
        confidence = np.ones((t, 6, 8), np.float32)
        motion = MotionSequence(fields, fields.copy(), confidence, confidence.copy())
        sync = DepthSync(
            DepthSyncConfig(
                algorithm_version="v3",
                min_fit_pixels=32,
                sample_count=256,
                residual_grid_shape=(9, 12),
            )
        )
        result = sync.offline_prepare(video, photo, anchor, motion=motion)
        self.assertGreater(float(np.mean(result.static_mask)), 0.6)
        wall_median = np.median(result.depths[:, :, : w // 3], axis=(1, 2))
        self.assertLess(float(np.ptp(wall_median)), 1e-4)
        replay = sync.apply_frame(video[0], result.parameters[0])
        np.testing.assert_allclose(replay, result.depths[0], atol=1e-6)

    def test_v4_uses_compact_region_parameters_without_photo_residual(self):
        t, h, w, anchor = 7, 36, 48, 3
        video_anchor = np.full((h, w), 0.25, np.float32)
        video_anchor[:, w // 2 :] = 0.45
        video_anchor[8:30, 19:31] = 0.85
        video = np.stack(
            [video_anchor + 0.005 * (index - anchor) for index in range(t)]
        )
        photo = np.full((h, w), 0.30, np.float32)
        photo[:, w // 2 :] = 0.55
        photo[8:30, 19:31] = 0.95
        fields = np.zeros((t, 6, 8, 2), np.float32)
        confidence = np.ones((t, 6, 8), np.float32)
        sync = DepthSync(
            DepthSyncConfig(
                algorithm_version="v4",
                min_fit_pixels=32,
                sample_count=256,
                region_grid_shape=(h, w),
                region_min_area_fraction=0.02,
                region_edge_kernel=1,
            )
        )
        result = sync.offline_prepare(
            video,
            photo,
            anchor,
            motion=MotionSequence(fields, fields.copy(), confidence, confidence.copy()),
            face_box=(19 / w, 8 / h, 31 / w, 30 / h),
        )
        self.assertEqual(result.residual_grids.size, 0)
        self.assertEqual(result.static_mask.size, 0)
        self.assertEqual(result.static_target_grid.size, 0)
        self.assertGreater(int(np.max(result.region_labels)), 0)
        self.assertEqual(
            int(np.max(result.region_labels[8:30, 19:31])),
            0,
        )
        replay = sync.apply_frame(video[0], result.parameters[0])
        np.testing.assert_allclose(replay, result.depths[0], atol=1e-6)
        parameters = result.parameters[0]
        global_only = sync.apply_frame(
            video[0],
            FrameParameters(
                parameters.scale,
                parameters.offset,
                parameters.confidence,
                parameters.fallback_reason,
                parameters.lut_x,
                parameters.lut_y,
                guidance_range=parameters.guidance_range,
            ),
        )
        subject_delta = replay[10:28, 20:30] - global_only[10:28, 20:30]
        self.assertLess(float(np.max(np.abs(subject_delta))), 1e-6)

    def test_v4_region_shape_removes_thin_tail(self):
        mask = np.zeros((24, 32), bool)
        mask[3:21, 2:14] = True
        mask[17:19, 14:28] = True
        cleaned = _largest_open_component(mask, radius=2)
        self.assertTrue(bool(cleaned[10, 8]))
        self.assertFalse(bool(cleaned[18, 24]))

    def test_v4_region_correction_reaches_depth_boundary(self):
        depth = np.full((12, 16), 0.8, np.float32)
        depth[:, :8] = 0.2
        labels = np.zeros((12, 16), np.uint8)
        labels[:, :8] = 1
        params = FrameParameters(
            1.0,
            0.0,
            1.0,
            guidance_range=0.6,
            region_labels=labels,
            region_scales=np.ones(2, np.float32),
            region_offsets=np.array([0.0, 0.1], np.float32),
            region_shift=np.zeros(2, np.float32),
        )
        output = DepthSync().apply_frame(depth, params)
        np.testing.assert_allclose(output[:, :8], 0.3, atol=1e-6)
        np.testing.assert_allclose(output[:, 8:], 0.8, atol=1e-6)

    def test_v4_adaptive_lut_is_no_worse_than_fixed_candidates(self):
        h, w = 48, 72
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        anchor = 0.1 + 0.8 * xx / (w - 1) + 0.05 * yy / (h - 1)
        photo = 0.2 + 0.45 * anchor + 0.6 * anchor**2
        video = np.stack([anchor - 0.002, anchor, anchor + 0.002])
        errors = {}
        for candidates in ((8,), (16,), (8, 16)):
            result = DepthSync(
                DepthSyncConfig(
                    lut_nodes=16,
                    lut_candidate_nodes=candidates,
                    region_grid_shape=(0, 0),
                    subject_offset_clip_fraction=0.0,
                    min_fit_pixels=64,
                    sample_count=1024,
                )
            ).offline_prepare(video, photo, 1)
            errors[candidates] = float(
                np.median(np.abs(result.depths[1] - photo))
            )
        self.assertLessEqual(
            errors[(8, 16)],
            min(errors[(8,)], errors[(16,)]) + 1e-6,
        )

    def test_v41_recovers_large_nonlinear_range_with_invalid_floor(self):
        """A DAv2-like range mismatch must not silently become identity."""
        h, w = 72, 128
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        source = 18.0 * (0.75 * xx / (w - 1) + 0.25 * yy / (h - 1))
        source[:26] = 0.0  # Model invalid-floor plateau, as seen in scene 12.
        photo = 8.0 + 4.0 * source + 1.15 * source**2
        photo[:26] = 0.0
        video = np.stack([source - 0.02, source, source + 0.02])
        result = DepthSync(
            DepthSyncConfig(
                algorithm_version="v4",
                lut_nodes=64,
                lut_candidate_nodes=(8, 16, 32, 64),
                sample_count=16384,
                subject_offset_clip_fraction=0.0,
            )
        ).offline_prepare(video, photo, 1)
        valid = source > 0.0
        raw_error = float(np.median(np.abs(source[valid] - photo[valid])))
        mapped_error = float(
            np.median(np.abs(result.depths[1][valid] - photo[valid]))
        )
        self.assertLess(mapped_error, raw_error * 0.08)
        self.assertGreater(float(result.lut_y[1, -1]), 200.0)
        self.assertGreater(float(result.confidences[1]), 0.0)
        self.assertNotEqual(result.fallback_reasons[1], "insufficient_anchor_support")

    def test_v41_quantile_fallback_aligns_distribution_when_pairs_are_wrong(self):
        """Global range still aligns when spatial pairing is unreliable."""
        h, w = 64, 96
        source = np.tile(np.linspace(0.2, 10.0, w, dtype=np.float32), (h, 1))
        target_distribution = 20.0 + 3.0 * source + 0.8 * source**2
        photo = np.flip(target_distribution, axis=1).copy()  # Deliberately wrong pairs.
        video = np.stack([source, source, source])
        result = DepthSync(
            DepthSyncConfig(
                algorithm_version="v4",
                lut_nodes=64,
                lut_candidate_nodes=(8, 16, 32, 64),
                sample_count=16384,
                subject_offset_clip_fraction=0.0,
            )
        ).offline_prepare(video, photo, 1)
        before = np.quantile(source, [0.1, 0.5, 0.9])
        after = np.quantile(result.depths[1], [0.1, 0.5, 0.9])
        target = np.quantile(photo, [0.1, 0.5, 0.9])
        self.assertLess(float(np.mean(np.abs(after - target))), float(np.mean(np.abs(before - target))) * 0.1)
        self.assertTrue(np.all(np.diff(result.lut_y[1]) >= -1e-6))

    def test_v41_distribution_stabilization_reduces_global_scale_drift(self):
        h, w, t, anchor = 48, 80, 9, 4
        base = np.tile(np.linspace(0.2, 8.0, w, dtype=np.float32), (h, 1))
        scales = np.linspace(0.7, 1.3, t, dtype=np.float32)
        video = np.stack([base * scale for scale in scales])
        photo = 10.0 + 5.0 * base + base**2
        config = dict(
            algorithm_version="v4",
            lut_nodes=64,
            lut_candidate_nodes=(8, 16, 32, 64),
            sample_count=16384,
            subject_offset_clip_fraction=0.0,
        )
        fixed = DepthSync(
            DepthSyncConfig(
                **config,
                enable_lut_distribution_stabilization=False,
            )
        ).offline_prepare(video, photo, anchor)
        stabilized = DepthSync(DepthSyncConfig(**config)).offline_prepare(
            video, photo, anchor
        )
        fixed_medians = np.median(fixed.depths, axis=(1, 2))
        stabilized_medians = np.median(stabilized.depths, axis=(1, 2))
        self.assertLess(
            float(np.ptp(stabilized_medians)),
            float(np.ptp(fixed_medians)) * 0.55,
        )
        self.assertTrue(np.all(stabilized.confidences > 0.0))

    def test_isolated_static_confidence_hole_is_filled(self):
        t, h, w = 7, 30, 40
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        photo = 0.2 + 0.3 * xx / (w - 1) + 0.1 * yy / (h - 1)
        video = np.stack([photo + 0.01 * i for i in range(t)])
        fields = np.zeros((t, 6, 8, 2), np.float32)
        confidence = np.ones((t, 6, 8), np.float32)
        confidence[:, 3, 4] = 0.0
        config = DepthSyncConfig(
            algorithm_version="v3",
            min_fit_pixels=32,
            sample_count=256,
            static_grid_shape=(6, 8),
            static_close_radius=1,
            static_erode_radius=0,
            static_mask_blur_sigma=0.0,
            static_edge_threshold_fraction=10.0,
        )
        result = DepthSync(config).offline_prepare(
            video,
            photo,
            3,
            motion=MotionSequence(fields, fields.copy(), confidence, confidence.copy()),
        )
        self.assertGreater(float(result.static_mask[3, 4]), 0.9)

    def test_face_foreground_prior_remains_a_hard_static_barrier(self):
        t, h, w = 7, 36, 48
        photo = np.full((h, w), 0.2, np.float32)
        photo[10:26, 18:33] = 0.9
        video = np.repeat(photo[None], t, axis=0)
        fields = np.zeros((t, 6, 8, 2), np.float32)
        confidence = np.ones((t, 6, 8), np.float32)
        config = DepthSyncConfig(
            algorithm_version="v3",
            min_fit_pixels=32,
            sample_count=256,
            static_grid_shape=(18, 24),
            static_target_shape=(18, 24),
            static_mask_blur_sigma=0.0,
        )
        result = DepthSync(config).offline_prepare(
            video,
            photo,
            3,
            motion=MotionSequence(fields, fields.copy(), confidence, confidence.copy()),
            face_box=(0.30, 0.15, 0.75, 0.85),
        )
        self.assertLess(float(np.max(result.static_mask[5:13, 9:16])), 1e-6)
        self.assertGreater(float(np.mean(result.static_mask[:, :6])), 0.9)

    def test_subject_track_rejects_whole_connected_depth_plane(self):
        t, h, w = 7, 18, 24
        photo = np.full((h, w), 0.2, np.float32)
        photo[:, 12:] = 0.6
        sequence = np.repeat(photo[None], t, axis=0)
        sequence[:3, 6:14, 15:21] = 0.9
        fields = np.zeros((t, h, w, 2), np.float32)
        fields[1:4, 6:14, 15:21, 0] = 0.02
        confidence = np.ones((t, h, w), np.float32)
        config = DepthSyncConfig(
            static_grid_shape=(h, w),
            static_close_radius=0,
            static_mask_blur_sigma=0.0,
            static_expand_radius=0,
            static_subject_box_margin=1,
        )
        mask, subject_barrier = _static_guidance_mask(
            MotionSequence(fields, fields.copy(), confidence, confidence.copy()),
            list(sequence),
            photo,
            0.7,
            (h, w),
            config,
            face_box=(0.65, 0.30, 0.90, 0.80),
        )
        self.assertGreater(float(np.mean(mask[:, :10])), 0.9)
        self.assertLess(float(np.max(mask[:, 12:])), 1e-6)
        self.assertGreater(float(np.mean(subject_barrier[5:15, 14:22])), 0.5)

    def test_apply_frame_uses_only_scale_and_offset(self):
        depth = np.array([[1.0, 2.0]], np.float32)
        result = DepthSync().apply_frame(depth, FrameParameters(2.0, -0.5, 1.0))
        np.testing.assert_allclose(result, [[1.5, 3.5]])

    def test_coarse_correction_is_suppressed_at_depth_edges(self):
        depth = np.full((12, 16), 0.2, np.float32)
        depth[:, 8:] = 0.8
        params = FrameParameters(
            1.0,
            0.0,
            1.0,
            residual_grid=np.full((3, 4), 0.3, np.float32),
            guidance_range=0.6,
        )
        output = DepthSync().apply_frame(depth, params)
        self.assertGreater(float(output[6, 1] - depth[6, 1]), 0.25)
        self.assertLess(float(abs(output[6, 7] - depth[6, 7])), 1e-5)

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
