import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_brow_pseudolabel import (  # noqa: E402
    CANDIDATES,
    OUTPUT_SCHEMA,
    draw_blind_candidate,
    make_contact_sheet,
    validate_judgment,
)


def valid_judgment():
    return {
        "scores": [
            {
                "option_id": option,
                "compatibility_score": score,
                "naturalness_score": score - 2,
                "proportion_fit_score": score - 1,
                "notes": "Test note grounded in the candidate preview.",
            }
            for option, score in (("A", 70), ("B", 80), ("C", 75))
        ],
        "uncertainty": "Synthetic test response only.",
    }


class PseudoLabelTests(unittest.TestCase):
    def test_schema_has_explicit_three_option_output(self):
        self.assertEqual(OUTPUT_SCHEMA["properties"]["scores"]["minItems"], 3)
        self.assertEqual(OUTPUT_SCHEMA["properties"]["scores"]["maxItems"], 3)

    def test_validate_judgment_accepts_exactly_a_b_c(self):
        result = validate_judgment(valid_judgment())
        self.assertEqual([row["option_id"] for row in result["scores"]], ["A", "B", "C"])

    def test_validate_judgment_rejects_duplicate_option(self):
        data = valid_judgment()
        data["scores"][2]["option_id"] = "B"
        with self.assertRaises(ValueError):
            validate_judgment(data)

    def test_validate_judgment_rejects_out_of_range_scores(self):
        data = valid_judgment()
        data["scores"][0]["compatibility_score"] = 101
        with self.assertRaises(ValueError):
            validate_judgment(data)

    def test_draw_and_contact_sheet_make_visible_three_candidate_panels(self):
        h = w = 300
        image = np.full((h, w, 3), 180, dtype=np.uint8)
        points = np.zeros((468, 3), dtype=np.float64)
        for i in range(len(points)):
            points[i] = (40 + (i % 30) * 7, 30 + (i % 35) * 6, 0)
        points[33, :2] = (95, 130)
        points[263, :2] = (205, 130)
        for ids in ((70, 63, 105, 66, 107), (300, 293, 334, 296, 336)):
            for j, idx in enumerate(ids):
                points[idx, :2] = (70 + j * 15, 105 - abs(2-j)*3)
        panel = draw_blind_candidate(image, points, CANDIDATES[0], "A")
        contact = make_contact_sheet(image, points, list(CANDIDATES), ["A", "B", "C"])
        self.assertGreater(panel.shape[0], 0)
        self.assertEqual(contact.shape[1], panel.shape[1] * 3)
        self.assertGreater(contact.shape[0], panel.shape[0] - 100)


if __name__ == "__main__":
    unittest.main()
