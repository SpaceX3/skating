import unittest

import torch

from action_rag import EvidenceRAG, masked_softmax
from model import scoring_head


def synthetic_corpus(key_dim=6):
    keys = torch.randn(5, key_dim)
    return {
        "keys": keys,
        "video_ids": ["v0", "v0", "v1", "v2", "v3"],
        "instance_ids": ["i0", "i1", "i2", "i3", "i4"],
        "coarse_class_ids": torch.tensor([2, 3, 4, 1, 2]),
        "element_ids": torch.tensor([1, 2, 3, 0, 1]),
        "elements": ["3A", "CCOSP4", "STSQ3", "<background>", "3A"],
        "goe_grades": torch.tensor([2.0, -1.0, 1.0, 0.0, 3.0]),
        "goe_points": torch.tensor([1.6, -0.3, 0.4, 0.0, 2.4]),
        "bvs": torch.tensor([8.0, 3.5, 3.3, 0.0, 8.0]),
        "panel_scores": torch.tensor([9.6, 3.2, 3.7, 0.0, 10.4]),
        "valid_score_mask": torch.tensor([True, True, True, False, True]),
        "coarse_class_vocab": ["unknown", "background", "jump", "spin", "sequence"],
        "element_vocab": ["<unknown>", "3A", "CCOSP4", "STSQ3"],
        "metadata_version": "test-v1",
    }


class MaskedSoftmaxTest(unittest.TestCase):
    def test_all_invalid_row_is_zero(self):
        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[True, False], [False, False]])
        weights = masked_softmax(logits, mask)
        self.assertTrue(torch.allclose(weights[0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(weights[1], torch.zeros(2)))


class EvidenceRAGTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.module = EvidenceRAG(
            synthetic_corpus(),
            dynamic_dim=8,
            query_dim=6,
            evidence_dim=8,
            metadata_dim=4,
            delta_max=10.0,
        )
        with torch.no_grad():
            self.module.correction_head[-1].bias.fill_(0.5)

    def inputs(self):
        return {
            "dynamic_time_feat": torch.randn(2, 3, 8),
            "static_raw": torch.randn(2, 3, 6),
            "dynamic_valid_mask": torch.tensor(
                [[True, True, True], [True, True, False]]
            ),
            "static_valid_mask": torch.tensor(
                [[True, True, True], [True, True, False]]
            ),
            "candidate_indices": torch.tensor(
                [
                    [[0, 1], [2, 3], [4, -1]],
                    [[3, -1], [3, -1], [-1, -1]],
                ]
            ),
            "candidate_similarities": torch.tensor(
                [
                    [[0.9, 0.8], [0.7, 0.6], [0.5, 0.0]],
                    [[0.9, 0.0], [0.8, 0.0], [0.0, 0.0]],
                ]
            ),
            "overlap_weights": torch.tensor(
                [
                    [[2.0, 0.0, 0.0], [0.5, 1.5, 0.0], [0.0, 0.5, 1.5]],
                    [[2.0, 0.0, 0.0], [0.5, 1.5, 0.0], [0.0, 0.0, 0.0]],
                ]
            ),
        }

    def test_no_scored_reference_forces_zero_delta(self):
        output = self.module(**self.inputs())
        self.assertNotEqual(float(output["delta_tes_rag"][0]), 0.0)
        self.assertEqual(float(output["delta_tes_rag"][1]), 0.0)
        self.assertEqual(float(output["evidence_valid_mask"][1]), 0.0)
        self.assertTrue(torch.isfinite(output["citation_weights"]).all())

    def test_padding_has_zero_citation_weight(self):
        inputs = self.inputs()
        output = self.module(**inputs)
        invalid = inputs["candidate_indices"].lt(0)
        self.assertTrue(torch.equal(output["citation_weights"][invalid], torch.zeros_like(output["citation_weights"][invalid])))

    def test_corpus_keys_do_not_receive_gradients(self):
        output = self.module(**self.inputs())
        output["delta_tes_rag"].sum().backward()
        self.assertIsNone(self.module.corpus_keys.grad)
        self.assertIsNotNone(self.module.reference_encoder[1].weight.grad)


class FullModelTest(unittest.TestCase):
    def test_batch_steps_use_maximum_per_sample_shared_length(self):
        torch.manual_seed(4)
        model = scoring_head(
            depth=1,
            input_dim=4,
            dim=8,
            input_len=2,
            use_static_baseline=True,
            static_in_dim=6,
            baseline_static_proj_dim=4,
        )
        batch, padded_steps, shared_steps = 2, 5, 3
        audio = torch.randn(batch, padded_steps, 1, 4)
        video = torch.randn(batch, padded_steps, 1, 4)
        output = model(
            audio,
            video,
            torch.flip(audio, [1]),
            torch.flip(video, [1]),
            [5, 3],
            [3, 5],
            static_feature=torch.randn(batch, shared_steps, 6),
            static_valid_mask=torch.ones(batch, shared_steps, dtype=torch.bool),
        )
        self.assertEqual(output["dynamic_time_feat"].shape[1], shared_steps)
        self.assertTrue(output["dynamic_valid_mask"].all())

    def test_zero_initialized_correction_equals_metric_matched_baseline(self):
        torch.manual_seed(9)
        model = scoring_head(
            depth=1,
            input_dim=4,
            dim=8,
            input_len=3,
            use_static_branch=True,
            use_static_baseline=True,
            static_in_dim=6,
            static_proj_dim=8,
            baseline_static_proj_dim=4,
            rag_corpus=synthetic_corpus(),
        )
        batch, steps = 2, 3
        audio = torch.randn(batch, steps, 1, 4)
        video = torch.randn(batch, steps, 2, 4)
        output = model(
            audio,
            video,
            torch.flip(audio, [1]),
            torch.flip(video, [1]),
            [3, 2],
            [3, 2],
            static_feature=torch.randn(batch, steps, 6),
            static_valid_mask=torch.tensor(
                [[True, True, True], [True, True, False]]
            ),
            candidate_indices=torch.tensor(
                [
                    [[0, 1], [2, 3], [4, -1]],
                    [[0, 3], [1, 3], [-1, -1]],
                ]
            ),
            candidate_similarities=torch.rand(batch, steps, 2),
            overlap_weights=torch.tensor(
                [
                    [[2.0, 0.0, 0.0], [0.5, 1.5, 0.0], [0.0, 0.5, 1.5]],
                    [[2.0, 0.0, 0.0], [0.5, 1.5, 0.0], [0.0, 0.0, 0.0]],
                ]
            ),
        )
        self.assertTrue(torch.allclose(output["score"], output["tes_baseline"]))
        self.assertTrue(
            torch.equal(output["tes_dynamic"], output["tes_baseline"])
        )
        self.assertTrue(torch.equal(output["delta_tes_rag"], torch.zeros(batch)))
        self.assertTrue(torch.isfinite(output["expected_element_embedding"]).all())
        self.assertTrue(torch.isfinite(output["evidence_delta_goe_grade"]).all())
        self.assertEqual(output["evidence_reference_goe_grade"].shape, (batch, steps))
        self.assertTrue(
            torch.equal(output["dynamic_valid_mask"][1], torch.tensor([True, True, False]))
        )


if __name__ == "__main__":
    unittest.main()
