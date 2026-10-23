"""Shared area gate for candidate eyebrow masks.

Keep this numeric rule aligned with BrowSegmentationEngine.kt.
A candidate outside the plausible ROI fraction is rejected for manual correction.
The 2% lower bound and outer-two-pixel border rule intentionally reject sparse or leaking masks observed in public smoke tests.
"""

MIN_BROW_MASK_FRACTION = 0.02
MAX_BROW_MASK_FRACTION = 0.28


def is_plausible_brow_mask(
    selected_pixels: int,
    region_pixels: int,
    border_pixels: int = 0,
) -> bool:
    if (
        region_pixels <= 0
        or selected_pixels < 0
        or selected_pixels > region_pixels
        or border_pixels < 0
        or border_pixels > selected_pixels
    ):
        return False
    fraction = selected_pixels / region_pixels
    if not MIN_BROW_MASK_FRACTION <= fraction <= MAX_BROW_MASK_FRACTION:
        return False
    # Keep the edge-contact rule identical to BrowSegmentationEngine.kt.
    return border_pixels <= max(2, int(selected_pixels * 0.01))
