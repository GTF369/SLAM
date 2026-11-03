import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from split_choice_cases import read_and_validate, split_rows


def fake_rows(cases=20):
    rows = []
    styles = [("natural", False), ("soft", False), ("defined", True)]
    for case in range(cases):
        session = f"session-{case}"
        for style, selected in styles:
            rows.append({
                "case_id": f"customer-{case}",
                "session_id": session,
                "style": style,
                "selected": str(selected).lower(),
                "rating": "5" if selected else "",
                "consent_recorded": "true",
                "consent_version": "brow-choice-v1",
                "consent_timestamp": str(1800000000000 + case),
                "display_position": str({"natural": 0, "soft": 1, "defined": 2}[style]),
                "design_intent": "均衡自然",
            })
    return rows


class SplitChoiceCasesTests(unittest.TestCase):
    def test_keeps_every_customer_wholly_inside_one_split(self):
        rows = fake_rows(30)
        parts = split_rows(rows, seed=23)
        customer_sets = {
            name: {row["case_id"] for row in split}
            for name, split in parts.items()
        }
        self.assertEqual(set.union(*customer_sets.values()), {f"customer-{i}" for i in range(30)})
        names = list(customer_sets)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                self.assertFalse(customer_sets[left] & customer_sets[right])
        for split in parts.values():
            self.assertEqual(len(split) % 3, 0)

    def test_is_deterministic_for_fixed_seed(self):
        rows = fake_rows(25)
        left = split_rows(rows, seed=2026)
        right = split_rows(rows, seed=2026)
        for name in left:
            self.assertEqual(
                [row["case_id"] for row in left[name]],
                [row["case_id"] for row in right[name]],
            )

    def test_rejects_incomplete_or_ambiguous_session(self):
        rows = fake_rows(4)
        with self.assertRaises(ValueError):
            split_rows(rows[:-1])

        ambiguous = fake_rows(4)
        ambiguous[0]["selected"] = "true"
        with self.assertRaises(ValueError):
            split_rows(ambiguous)

    def test_rejects_unconsented_rows(self):
        rows = fake_rows(4)
        rows[0]["consent_recorded"] = "false"
        with self.assertRaisesRegex(ValueError, "without recorded consent"):
            split_rows(rows)

    def test_rejects_duplicate_display_positions(self):
        rows = fake_rows(4)
        rows[1]["display_position"] = "0"
        with self.assertRaisesRegex(ValueError, "display position"):
            split_rows(rows)

    def test_rejects_mixed_design_intent_within_session(self):
        rows = fake_rows(4)
        rows[2]["design_intent"] = "偏立体"
        with self.assertRaisesRegex(ValueError, "consistent"):
            split_rows(rows)

    def test_reads_valid_csv_and_requires_stable_case_and_session(self):
        rows = fake_rows(5)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "valid.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            headers, read_rows_back = read_and_validate(path)
            self.assertIn("session_id", headers)
            self.assertEqual(len(read_rows_back), 15)


if __name__ == "__main__":
    unittest.main()
