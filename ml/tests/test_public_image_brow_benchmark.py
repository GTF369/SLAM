import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from public_image_brow_benchmark import (
    CANDIDATES,
    candidate_brow_curve,
    circular_arc_factor,
    face_measurements,
    reference_weights,
    relative_depth_factor,
    score_candidates,
)


class PublicImageBrowBenchmarkTests(unittest.TestCase):
    def test_reference_weights_sum_to_one_but_are_not_empirical_probabilities(self):
        weights = reference_weights([82.0, 79.0, 75.0])
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertFalse(np.allclose(weights, np.full(3, 1.0 / 3.0)))

    def test_arc_factor_increases_with_curvature(self):
        shallow = circular_arc_factor(5.0, 100.0)
        curved = circular_arc_factor(5.0, 20.0)
        self.assertGreater(shallow, 1.0)
        self.assertGreater(curved, shallow)

    def test_arc_factor_rejects_impossible_chord(self):
        with self.assertRaises(ValueError):
            circular_arc_factor(20.0, 10.0)

    def test_relative_depth_factor_is_clamped_and_never_below_one(self):
        self.assertAlmostEqual(relative_depth_factor(10.0, 0.0), 1.0)
        self.assertAlmostEqual(relative_depth_factor(10.0, 100.0), 1.4)
        self.assertTrue(np.isnan(relative_depth_factor(0.0, 1.0)))

    def test_brow_curve_is_finite_and_has_many_smooth_samples(self):
        points = np.zeros((478, 3), dtype=np.float64)
        points[70] = (100, 100, 0)
        points[63] = (110, 90, 0)
        points[105] = (120, 82, 0)
        points[66] = (130, 89, 0)
        points[107] = (140, 101, 0)
        curve = candidate_brow_curve(points, (70, 63, 105, 66, 107), length_factor=1.0, arch_factor=1.1)
        self.assertEqual(curve.ndim, 2)
        self.assertEqual(curve.shape[1], 2)
        self.assertTrue(np.isfinite(curve).all())
        self.assertGreater(curve.shape[0], 20)

    def test_app_compatible_reference_scoring_returns_three_styles(self):
        points = np.zeros((478, 3), dtype=np.float64)
        points[234, :2] = (100.0, 120.0)
        points[454, :2] = (300.0, 120.0)
        points[10, :2] = (200.0, 20.0)
        points[152, :2] = (200.0, 260.0)
        points[33, :2] = (155.0, 105.0)
        points[263, :2] = (245.0, 108.0)
        points[172, :2] = (135.0, 200.0)
        points[397, :2] = (265.0, 200.0)
        points[105, :2] = (165.0, 91.0)
        points[159, :2] = (165.0, 104.0)
        points[334, :2] = (235.0, 94.0)
        points[386, :2] = (235.0, 106.0)
        features = face_measurements(points)
        scores = score_candidates(features)
        self.assertEqual({row["style"] for row in scores}, {row["style"] for row in CANDIDATES})
        self.assertAlmostEqual(sum(row["normalized_reference_weight_temperature_5"] for row in scores), 1.0, places=12)
        self.assertTrue(all(row["geometry_reference_score"] > 0 for row in scores))


if __name__ == "__main__":
    unittest.main()
