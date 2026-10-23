"""Unit tests for the fail-closed eyebrow mask area gate."""

import unittest

from ml.brow_mask_quality import is_plausible_brow_mask


class BrowMaskQualityTests(unittest.TestCase):
    def test_accepts_typical_eyebrow_area(self):
        self.assertTrue(is_plausible_brow_mask(100, 1000))
        self.assertTrue(is_plausible_brow_mask(5, 1000) is False)
        self.assertTrue(is_plausible_brow_mask(5, 1000) is False)

    def test_rejects_oversized_fallback_mask(self):
        # The public-image smoke report observed a 33.29% ROI mask on a blurred photo.
        self.assertFalse(is_plausible_brow_mask(3329, 10000))
        self.assertFalse(is_plausible_brow_mask(281, 1000))

    def test_accepts_documented_boundaries(self):
        self.assertTrue(is_plausible_brow_mask(5, 1000))
        self.assertTrue(is_plausible_brow_mask(280, 1000))

    def test_rejects_empty_and_invalid_counts(self):
        self.assertFalse(is_plausible_brow_mask(0, 1000))
        self.assertFalse(is_plausible_brow_mask(1, 1000))
        self.assertFalse(is_plausible_brow_mask(1001, 1000))
        self.assertFalse(is_plausible_brow_mask(0, 0))
        self.assertFalse(is_plausible_brow_mask(-1, 1000))


if __name__ == "__main__":
    unittest.main()
