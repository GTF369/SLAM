"""Shared area gate for candidate eyebrow masks.

Keep this numeric rule aligned with BrowSegmentationEngine.kt.
A candidate outside the plausible ROI fraction is rejected for manual correction.
The 2% lower bound intentionally rejects sparse, speck-like masks observed in public smoke tests.
"""

MIN_BROW_MASK_FRACTION = 0.02
MAX_BROW_MASK_FRACTION = 0.28


def is_plausible_brow_mask(selected_pixels: int, region_pixels: int) -> bool:
    if region_pixels <= 0 or selected_pixels < 0 or selected_pixels > region_pixels:
        return False
    fraction = selected_pixels / region_pixels
    return MIN_BROW_MASK_FRACTION <= fraction <= MAX_BROW_MASK_FRACTION
