import unittest

import numpy as np

from routing_ablation import random_one_class_probabilities


class RandomClassRoutingTests(unittest.TestCase):
    def test_routing_is_reproducible_and_one_hot(self):
        first = random_one_class_probabilities(64, 4, 2026, 3)
        second = random_one_class_probabilities(64, 4, 2026, 3)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (64, 4))
        np.testing.assert_array_equal(first.sum(axis=1), np.ones(64, dtype=np.float32))
        self.assertTrue(np.isin(first, [0.0, 1.0]).all())
        selected = np.argmax(first, axis=1)
        self.assertGreaterEqual(int(selected.min()), 0)
        self.assertLess(int(selected.max()), 4)

    def test_invalid_sizes_are_rejected(self):
        for num_queries, num_classes in ((0, 4), (64, 0), (-1, 4), (64, -1)):
            with self.assertRaises(ValueError):
                random_one_class_probabilities(
                    num_queries, num_classes, 2026, 3
                )


if __name__ == "__main__":
    unittest.main()
