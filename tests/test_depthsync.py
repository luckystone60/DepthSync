import unittest

import cv2
import numpy as np

from depthsync import DepthSync, DepthSyncConfig


class DepthSyncTest(unittest.TestCase):
    def test_anchor_scale_and_temporal_jump_are_improved(self):
        h, w, t, anchor = 72, 96, 7, 3
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        truth = 1.5 + 0.004 * xx + 0.002 * yy + 0.25 * (np.hypot(xx - 48, yy - 36) < 18)
        photo = truth + 0.025 * np.sin(xx * 0.8) * np.sin(yy * 0.7)
        low = cv2.resize(truth, (48, 36), interpolation=cv2.INTER_AREA)
        video = np.stack([(low - 0.12 * i) / (0.72 + 0.07 * i) for i in range(t)])
        result = DepthSync(DepthSyncConfig(depth_mode="depth", min_fit_pixels=64))(video, photo, anchor)
        before_anchor = cv2.resize(video[anchor], (w, h))
        before_error = np.mean(np.abs(before_anchor - photo))
        after_error = np.mean(np.abs(result.depths[anchor] - photo))
        self.assertLess(after_error, before_error * 0.35)
        before_jump = np.mean(np.abs(np.diff(np.stack([cv2.resize(x, (w, h)) for x in video]), axis=0)))
        after_jump = np.mean(np.abs(np.diff(result.depths, axis=0)))
        self.assertLess(after_jump, before_jump)
        self.assertEqual(result.depths.shape, (t, h, w))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            DepthSync()([], np.ones((4, 4), np.float32), 0)


if __name__ == "__main__":
    unittest.main()
