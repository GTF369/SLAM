import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_preference_cases import split_by_case


def rows_for_cases(count=40):
    rows = []
    for i in range(count):
        # Each synthetic case has both a high-rated and a low-rated candidate.
        rows.append({"case_id": f"SYN-{i:03d}", "rating": "5", "style": "natural"})
        rows.append({"case_id": f"SYN-{i:03d}", "rating": "2", "style": "soft"})
    return rows


class GroupedSplitTests(unittest.TestCase):
    def test_all_rows_for_a_case_stay_together_and_sets_are_disjoint(self):
        splits = split_by_case(rows_for_cases(), seed=7)
        ids = {name: {row["case_id"] for row in values} for name, values in splits.items()}
        self.assertTrue(ids["development"].isdisjoint(ids["calibration"]))
        self.assertTrue(ids["development"].isdisjoint(ids["external_validation"]))
        self.assertTrue(ids["calibration"].isdisjoint(ids["external_validation"]))
        for values in splits.values():
            grouped = {}
            for row in values:
                grouped.setdefault(row["case_id"], set()).add(row["rating"])
            self.assertTrue(all(ratings == {"2", "5"} for ratings in grouped.values()))

    def test_blank_case_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty case_id"):
            split_by_case([{"case_id": "", "rating": "5"}])

    def test_split_rejects_one_class_only(self):
        rows = [{"case_id": f"C{i}", "rating": "5"} for i in range(40)]
        with self.assertRaisesRegex(ValueError, "does not contain both rating classes"):
            split_by_case(rows)


if __name__ == "__main__":
    unittest.main()
