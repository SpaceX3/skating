import unittest

from finefs_class_bank import CLASS_NAMES, coarse_class_index


class FineFSClassBankTests(unittest.TestCase):
    def test_maps_dense_labels_to_c1_class_order(self):
        self.assertEqual(CLASS_NAMES, ("background", "jump", "spin", "sequence"))
        self.assertEqual(coarse_class_index("background"), 0)
        self.assertEqual(coarse_class_index("jump_1"), 1)
        self.assertEqual(coarse_class_index("jump_4"), 1)
        self.assertEqual(coarse_class_index("spin"), 2)
        self.assertEqual(coarse_class_index("sequence"), 3)


if __name__ == "__main__":
    unittest.main()
