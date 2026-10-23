import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_choice_sessions import split_choice_rows


def make_rows(customer_count=120):
    rows = []
    for i in range(customer_count):
        chosen = i % 3
        session = f"SESSION-{i:04d}"
        for style_index, style in enumerate(("natural", "soft", "defined")):
            rows.append({
                "case_id": f"CUSTOMER-{i:04d}",
                "session_id": session,
                "style": style,
                "thickness": str((1.0, 0.82, 1.18)[style_index]),
                "length": str((1.0, 0.94, 1.06)[style_index]),
                "arch": str((0.92, 0.72, 1.22)[style_index]),
                "selected": "true" if style_index == chosen else "false",
                "face_width_height_ratio": "0.72",
                "interocular_face_width_ratio": "0.40",
                "jaw_face_width_ratio": "0.75",
                "brow_eye_distance_ratio": "0.06",
                "brow_asymmetry_ratio": "0.01",
                "eye_line_tilt": "0.0",
                "native_brow_area_fraction": "0.10",
                "native_brow_side_imbalance": "0.02",
                "native_brow_mask_available": "1",
                "native_shape_mode": "0",
                "rating": str(2 + ((i + style_index) % 4)),
            })
    return rows


class ChoiceSessionSplitTests(unittest.TestCase):
    def test_three_styles_and_all_rows_for_a_customer_stay_in_one_split(self):
        splits = split_choice_rows(make_rows(), seed=13)
        ids = {name: {row["case_id"] for row in rows} for name, rows in splits.items()}
        self.assertTrue(ids["development"].isdisjoint(ids["calibration"]))
        self.assertTrue(ids["development"].isdisjoint(ids["external_validation"]))
        self.assertTrue(ids["calibration"].isdisjoint(ids["external_validation"]))
        for rows in splits.values():
            grouped = {}
            for row in rows:
                grouped.setdefault(row["session_id"], []).append(row)
            self.assertTrue(all(len(values) == 3 for values in grouped.values()))
            self.assertTrue(all(sum(r["selected"] == "true" for r in values) == 1 for values in grouped.values()))
            self.assertTrue(all({r["style"] for r in values} == {"natural", "soft", "defined"} for values in grouped.values()))

    def test_incomplete_triplet_is_rejected(self):
        rows = make_rows(120)
        rows.pop()
        with self.assertRaisesRegex(ValueError, "invalid choice sessions"):
            split_choice_rows(rows)


if __name__ == "__main__":
    unittest.main()
