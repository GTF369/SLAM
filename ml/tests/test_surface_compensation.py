"""Unit tests for measured 2D-to-surface arc length mapping."""
import math
import unittest

from ml.surface_compensation import (
    MAX_COMPENSATION_FACTOR,
    MIN_COMPENSATION_FACTOR,
    apply_segment_factors,
    compensation_factor_from_measurement,
    polyline_length_2d,
    polyline_length_3d,
    synthetic_surface_case,
)


class SurfaceCompensationTests(unittest.TestCase):
    def test_factor_uses_calibrated_projected_length_not_spatial_chord(self):
        factor = compensation_factor_from_measurement(
            surface_arc_mm=12.0,
            projected_segment_px=100.0,
            pupil_distance_px=630.0,
            interpupillary_distance_mm=63.0,
        )
        self.assertAlmostEqual(factor, 1.2, places=8)

    def test_rejects_nonphysical_or_unsupported_inputs(self):
        with self.assertRaises(ValueError):
            compensation_factor_from_measurement(0.0, 100.0, 600.0, 63.0)
        with self.assertRaises(ValueError):
            compensation_factor_from_measurement(12.0, 1.0, 600.0, 63.0)
        with self.assertRaises(ValueError):
            compensation_factor_from_measurement(100.0, 100.0, 600.0, 63.0)

    def test_segment_scaling_matches_each_target_length_and_recenters(self):
        points = [(0.0, 0.0), (10.0, 0.0), (20.0, 10.0)]
        adjusted = apply_segment_factors(points, [1.2, 0.8])
        self.assertAlmostEqual(polyline_length_2d(adjusted[:2]), 12.0, places=8)
        self.assertAlmostEqual(polyline_length_2d(adjusted[1:]), math.sqrt(200.0) * 0.8, places=8)
        self.assertAlmostEqual((adjusted[0][0] + adjusted[-1][0]) / 2, 10.0, places=8)
        self.assertAlmostEqual((adjusted[0][1] + adjusted[-1][1]) / 2, 5.0, places=8)

    def test_known_synthetic_face_surfaces_are_compensated(self):
        scenarios = [
            ("flat", 0.0, 0.0, 650.0),
            ("moderate", 3.0, 0.0, 600.0),
            ("strong", 8.0, 0.0, 600.0),
            ("asymmetric", 5.0, 1.5, 580.0),
        ]
        for name, bulge, asymmetry, camera_distance in scenarios:
            with self.subTest(name=name):
                result = synthetic_surface_case(
                    name=name,
                    frontal_bulge_mm=bulge,
                    asymmetry_mm=asymmetry,
                    camera_distance_mm=camera_distance,
                )
                self.assertLess(result["max_segment_error_mm"], 1e-8)
                self.assertLess(result["absolute_total_length_error_mm"], 1e-8)
                if result["raw_absolute_total_length_error_mm"] > 1e-3:
                    self.assertGreater(result["relative_total_error_reduction"], 0.999999)
                self.assertTrue(
                    all(MIN_COMPENSATION_FACTOR <= segment["factor"] <= MAX_COMPENSATION_FACTOR
                        for segment in result["segments"])
                )
                self.assertGreaterEqual(result["endpoint_shift_px"], 0.0)

    def test_3d_polyline_length_exceeds_planar_projection_for_dome(self):
        points3d = [(0.0, 0.0, 10.0), (10.0, 0.0, 12.0), (20.0, 0.0, 10.0)]
        self.assertGreater(polyline_length_3d(points3d), 20.0)


if __name__ == "__main__":
    unittest.main()
