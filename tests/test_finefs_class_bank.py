import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from finefs_class_bank import (
    CLASS_NAMES,
    SCORE_DIM,
    coarse_class_index,
    load_scored_intervals,
    match_score_descriptor,
)


class FineFSClassBankTests(unittest.TestCase):
    def test_maps_dense_labels_to_c1_class_order(self):
        self.assertEqual(CLASS_NAMES, ("background", "jump", "spin", "sequence"))
        self.assertEqual(coarse_class_index("background"), 0)
        self.assertEqual(coarse_class_index("jump_1"), 1)
        self.assertEqual(coarse_class_index("jump_4"), 1)
        self.assertEqual(coarse_class_index("spin"), 2)
        self.assertEqual(coarse_class_index("sequence"), 3)

    def test_action_interval_supplies_bv_goe_and_panel_score(self):
        payload = {
            "executed_element": {
                "element1": {
                    "element": "3A",
                    "coarse_class": "jump",
                    "bv": 8.0,
                    "goe": 1.6,
                    "score_of_pannel": 9.6,
                    "time": "0-22,0-25",
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            intervals = load_scored_intervals(path)
            score, valid = match_score_descriptor(
                intervals, start=21.5, end=22.5, coarse_class="jump"
            )

        self.assertEqual(SCORE_DIM, 3)
        self.assertTrue(valid)
        np.testing.assert_allclose(score, [8.0, 1.6, 9.6], atol=1e-6)

    def test_panel_typo_is_replaced_by_bv_plus_goe(self):
        payload = {
            "executed_element": {
                "element6": {
                    "element": "3A",
                    "coarse_class": "jump",
                    "bv": 13.42,
                    "goe": 2.97,
                    "score_of_pannel": 1639,
                    "time": "2-14,2-18",
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "898.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            intervals = load_scored_intervals(path)
            score, valid = match_score_descriptor(
                intervals, start=134.0, end=135.0, coarse_class="jump"
            )

        self.assertTrue(valid)
        np.testing.assert_allclose(score, [13.42, 2.97, 16.39], atol=1e-6)

    def test_background_has_no_element_score(self):
        score, valid = match_score_descriptor(
            (), start=0.0, end=1.0, coarse_class="background"
        )

        self.assertFalse(valid)
        np.testing.assert_array_equal(score, np.zeros(3, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
