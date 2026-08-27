import unittest

import numpy as np
import torch

import main
from class_conditioned_retrieval import pack_cross_attention_cache
from model import QuerySupportCrossAttention


class Top4CrossAttentionTests(unittest.TestCase):
    def test_cache_preserves_query_supports_and_class_weights(self):
        query = np.arange(6, dtype=np.float32).reshape(2, 3)
        supports = np.arange(2 * 2 * 4 * 3, dtype=np.float32).reshape(2, 2, 4, 3)
        weights = np.array([[0.75, 0.25], [0.4, 0.6]], dtype=np.float32)

        packed = pack_cross_attention_cache(query, supports, weights)

        self.assertEqual(packed.shape, (2, 29))
        np.testing.assert_array_equal(packed[:, :3], query)
        np.testing.assert_array_equal(packed[:, 3:27], supports.reshape(2, -1))
        np.testing.assert_array_equal(packed[:, -2:], weights)

    def test_model_reads_two_classes_of_four_supports(self):
        fusion = QuerySupportCrossAttention(
            feature_dim=8,
            top_classes=2,
            top_k=4,
            attention_dim=4,
            num_heads=2,
            dropout=0.0,
        )
        packed = torch.randn(3, 8 + 2 * 4 * 8 + 2)

        output = fusion(packed)

        self.assertEqual(tuple(output.shape), (3, 8))

    def test_training_model_projects_cross_attention_output(self):
        model = main.build_model(static_in_dim=6914)

        self.assertEqual(model.query_support_cross_attention.top_k, 4)
        self.assertEqual(model.query_support_cross_attention.top_classes, 2)
        self.assertEqual(model.static_proj.in_features, 768)


if __name__ == "__main__":
    unittest.main()
