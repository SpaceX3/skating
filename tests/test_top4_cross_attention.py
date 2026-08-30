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
        scores = np.arange(2 * 2 * 4 * 3, dtype=np.float32).reshape(2, 2, 4, 3)
        score_masks = np.ones((2, 2, 4), dtype=np.float32)
        weights = np.array([[0.75, 0.25], [0.4, 0.6]], dtype=np.float32)

        packed = pack_cross_attention_cache(
            query, supports, scores, score_masks, weights
        )

        self.assertEqual(packed.shape, (2, 61))
        np.testing.assert_array_equal(packed[:, :3], query)
        np.testing.assert_array_equal(packed[:, 3:27], supports.reshape(2, -1))
        np.testing.assert_array_equal(packed[:, 27:51], scores.reshape(2, -1))
        np.testing.assert_array_equal(packed[:, 51:59], score_masks.reshape(2, -1))
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
        packed = torch.randn(3, 8 + 2 * 4 * 8 + 2 * 4 * 3 + 2 * 4 + 2)

        output = fusion(packed)

        self.assertEqual(tuple(output.shape), (3, 8))

    def test_support_score_changes_fused_output(self):
        torch.manual_seed(7)
        fusion = QuerySupportCrossAttention(
            feature_dim=4,
            top_classes=1,
            top_k=2,
            score_dim=3,
            attention_dim=4,
            num_heads=1,
            dropout=0.0,
        ).eval()
        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        supports = np.array([[[[1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0]]]], dtype=np.float32)
        masks = np.ones((1, 1, 2), dtype=np.float32)
        weights = np.ones((1, 1), dtype=np.float32)
        low_scores = np.zeros((1, 1, 2, 3), dtype=np.float32)
        high_scores = low_scores.copy()
        high_scores[..., 2] = 2.0
        low = torch.from_numpy(
            pack_cross_attention_cache(query, supports, low_scores, masks, weights)
        )
        high = torch.from_numpy(
            pack_cross_attention_cache(query, supports, high_scores, masks, weights)
        )

        low_output = fusion(low)
        high_output = fusion(high)

        self.assertFalse(torch.allclose(low_output, high_output))

    def test_zero_support_score_variance_has_finite_gradients(self):
        torch.manual_seed(7)
        fusion = QuerySupportCrossAttention(
            feature_dim=4,
            top_classes=1,
            top_k=2,
            score_dim=3,
            attention_dim=4,
            num_heads=1,
            dropout=0.0,
        )
        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        supports = np.array(
            [[[[1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0]]]],
            dtype=np.float32,
        )
        identical_scores = np.ones((1, 1, 2, 3), dtype=np.float32)
        masks = np.ones((1, 1, 2), dtype=np.float32)
        weights = np.ones((1, 1), dtype=np.float32)
        packed = torch.from_numpy(
            pack_cross_attention_cache(
                query, supports, identical_scores, masks, weights
            )
        )

        fusion(packed).square().mean().backward()

        bad_gradients = [
            name
            for name, parameter in fusion.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        self.assertEqual(bad_gradients, [])

    def test_training_model_projects_cross_attention_output(self):
        model = main.build_model(static_in_dim=6946)

        self.assertEqual(model.query_support_cross_attention.top_k, 4)
        self.assertEqual(model.query_support_cross_attention.top_classes, 2)
        self.assertEqual(model.query_support_cross_attention.score_dim, 3)
        self.assertEqual(model.static_proj.in_features, 768)


if __name__ == "__main__":
    unittest.main()
