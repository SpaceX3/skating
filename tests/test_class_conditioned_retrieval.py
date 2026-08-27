import unittest

import numpy as np

from class_conditioned_retrieval import (
    ClassConditionedRetriever,
    aggregate_retrieved_vectors,
)


class ClassConditionedRetrievalTests(unittest.TestCase):
    def test_top2_probability_weighted_retrieval_fusion_preserves_query_and_shape(self):
        # One query, four coarse classes, two retrieved clips per class, 2-D toy features.
        retrieved = np.array(
            [
                [
                    [[1.0, 0.0], [0.0, 1.0]],  # background
                    [[2.0, 0.0], [2.0, 0.0]],  # jump
                    [[0.0, 3.0], [0.0, 3.0]],  # spin
                    [[4.0, 0.0], [4.0, 0.0]],  # sequence
                ]
            ],
            dtype=np.float32,
        )
        similarities = np.array(
            [[[0.1, 0.1], [0.9, 0.8], [0.2, 0.2], [0.7, 0.6]]],
            dtype=np.float32,
        )
        probabilities = np.array([[0.01, 0.60, 0.02, 0.37]], dtype=np.float32)
        query = np.array([[9.0, 8.0]], dtype=np.float32)

        fused, details = aggregate_retrieved_vectors(
            query,
            retrieved,
            similarities,
            probabilities,
            top_classes=2,
            temperature=0.1,
            probability_power=1.0,
        )

        self.assertEqual(fused.shape, (1, 4))
        np.testing.assert_allclose(fused[0, :2], query[0])
        # The retrieved representation is between jump [2, 0] and sequence [4, 0].
        self.assertGreater(fused[0, 2], 2.0)
        self.assertLess(fused[0, 2], 4.0)
        np.testing.assert_allclose(fused[0, 3], 0.0, atol=1e-6)
        self.assertEqual(details["selected_classes"].tolist(), [[1, 3]])

    def test_retriever_uses_only_routed_class_and_top_k_neighbors(self):
        banks = (
            np.array([[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 2.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32),
        )
        retriever = ClassConditionedRetriever(banks, device="cpu")

        fused, details = retriever.retrieve(
            query=np.array([[1.0, 0.0]], dtype=np.float32),
            class_probabilities=np.array([[0.9, 0.1]], dtype=np.float32),
            top_classes=1,
            top_k=2,
            temperature=0.1,
        )

        self.assertEqual(fused.shape, (1, 4))
        np.testing.assert_allclose(fused[0, :2], [1.0, 0.0])
        self.assertGreater(fused[0, 2], 0.8)
        self.assertGreater(fused[0, 3], 0.0)
        self.assertEqual(details["selected_classes"].tolist(), [[0]])


if __name__ == "__main__":
    unittest.main()
