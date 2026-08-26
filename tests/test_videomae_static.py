import unittest

import numpy as np
import torch

from videomae_static import (
    build_timestep_candidates,
    choose_by_ensemble_confidence,
    select_static_sequence,
)


class _FirstValueModel(torch.nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = float(scale)

    def forward(self, target_features, context_features):
        value = target_features[:, 0, 0] * self.scale
        zeros = torch.zeros_like(value)
        return torch.stack((value, zeros, zeros, zeros), dim=1)


class CandidateTests(unittest.TestCase):
    def test_uses_two_second_anchor_and_four_half_second_offsets(self):
        starts = np.arange(0.0, 12.5, 0.5)
        times = np.stack((starts, starts + 1.0), axis=1)

        candidates = build_timestep_candidates(times, timestep_index=2)

        self.assertEqual([item["start"] for item in candidates], [4.0, 4.5, 5.0, 5.5])
        self.assertTrue(all(len(item["target_indices"]) == 3 for item in candidates))
        self.assertTrue(all(len(item["context_indices"]) == 6 for item in candidates))
        np.testing.assert_array_equal(candidates[0]["target_indices"], [8, 9, 10])
        np.testing.assert_array_equal(candidates[0]["context_indices"], [11, 12, 13, 14, 15, 16])

    def test_tail_keeps_only_complete_five_second_candidates(self):
        starts = np.arange(0.0, 5.5, 0.5)
        times = np.stack((starts, starts + 1.0), axis=1)

        candidates = build_timestep_candidates(times, timestep_index=0)

        self.assertEqual([item["start"] for item in candidates], [0.0, 0.5, 1.0])


class ConfidenceTests(unittest.TestCase):
    def test_averages_seed_logits_then_selects_largest_max_softmax(self):
        seed_logits = np.asarray(
            [
                [[2.0, 0.0], [0.0, 3.0], [1.0, 0.0]],
                [[2.0, 0.0], [0.0, 1.0], [4.0, 0.0]],
            ],
            dtype=np.float64,
        )

        selected, probabilities = choose_by_ensemble_confidence(seed_logits)

        self.assertEqual(selected, 2)
        self.assertEqual(probabilities.shape, (3, 2))


class StaticSequenceTests(unittest.TestCase):
    def test_selects_candidate_and_takes_its_first_temporal_token(self):
        starts = np.arange(0.0, 8.5, 0.5)
        times = np.stack((starts, starts + 1.0), axis=1)
        features = np.zeros((len(starts), 8, 2), dtype=np.float32)
        for index in range(len(starts)):
            features[index, :, 0] = index + np.arange(8, dtype=np.float32)
            features[index, :, 1] = 100.0 + index

        sequence, report = select_static_sequence(
            features,
            times,
            dynamic_length=1,
            models=(_FirstValueModel(1.0), _FirstValueModel(0.5)),
            device="cpu",
            batch_size=2,
        )

        expected_first_index = 3
        np.testing.assert_allclose(
            sequence[0], features[expected_first_index, 0]
        )
        self.assertEqual(report["selected_offset_counts"]["1.5"], 1)
        self.assertEqual(report["previous_vector_fallbacks"], 0)

    def test_reuses_previous_vector_when_later_timestep_has_no_candidate(self):
        starts = np.arange(0.0, 5.0, 0.5)
        times = np.stack((starts, starts + 1.0), axis=1)
        features = np.arange(len(starts) * 8 * 2, dtype=np.float32).reshape(
            len(starts), 8, 2
        )

        sequence, report = select_static_sequence(
            features,
            times,
            dynamic_length=2,
            models=(_FirstValueModel(),),
            device="cpu",
        )

        np.testing.assert_allclose(sequence[1], sequence[0])
        self.assertEqual(report["previous_vector_fallbacks"], 1)


if __name__ == "__main__":
    unittest.main()
