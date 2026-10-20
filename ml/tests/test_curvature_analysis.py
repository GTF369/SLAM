import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_curvature import bootstrap_mean_ci, quantile


class CurvatureAnalysisTests(unittest.TestCase):
    def test_quantile_interpolates(self):
        self.assertAlmostEqual(quantile([1.0, 3.0], 0.25), 1.5)
        self.assertAlmostEqual(quantile([3.0, 1.0, 2.0], 0.5), 2.0)

    def test_single_value_interval(self):
        self.assertEqual(bootstrap_mean_ci([1.25]), (1.25, 1.25))

    def test_bootstrap_interval_contains_stable_mean(self):
        values = [1.01, 1.02, 1.00, 1.01, 1.02, 1.00]
        low, high = bootstrap_mean_ci(values, seed=42, draws=1000)
        mean = sum(values) / len(values)
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)


if __name__ == "__main__":
    unittest.main()
