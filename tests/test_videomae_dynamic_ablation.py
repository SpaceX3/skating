import unittest

import numpy as np

import main
from videomae_dynamic import build_dynamic_videomae_sequence


class VideoMAEDynamicTests(unittest.TestCase):
    def test_concatenates_five_one_second_offsets_as_40_tokens(self):
        starts = np.arange(0.0, 8.5, 0.5)
        times = np.stack((starts, starts + 1.0), axis=1)
        features = np.zeros((len(starts), 8, 768), dtype=np.float32)
        for index, start in enumerate(starts):
            features[index, :, 0] = start

        sequence = build_dynamic_videomae_sequence(
            features, times, dynamic_length=2
        )

        self.assertEqual(sequence.shape, (2, 40, 768))
        np.testing.assert_array_equal(
            sequence[0, :, 0].reshape(5, 8)[:, 0], [0, 1, 2, 3, 4]
        )
        np.testing.assert_array_equal(
            sequence[1, :, 0].reshape(5, 8)[:, 0], [2, 3, 4, 5, 6]
        )

    def test_static_branch_is_a_model_switch(self):
        dynamic_only = main.build_model(use_static_branch=False)
        with_static = main.build_model(use_static_branch=True)

        self.assertFalse(dynamic_only.use_static_branch)
        self.assertFalse(hasattr(dynamic_only, "query_support_cross_attention"))
        self.assertTrue(with_static.use_static_branch)
        self.assertEqual(with_static.query_support_cross_attention.top_k, 4)
        self.assertEqual(dynamic_only.linear_forward[0][0].fn[0].in_channels, 43)


if __name__ == "__main__":
    unittest.main()
