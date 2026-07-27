import unittest

import torch

from semantic_data import (
    FineFSSemanticDataset,
    SemanticCandidateSampler,
    make_video_split,
    validate_video_split,
)
from semantic_main import deduplicated_indices, gather, prepare_device_corpus
from scripts.build_action_rag_corpus import trimmed_judge_grade
from semantic_rag import FineFSSemanticRAG, multi_positive_nll, soft_target_nll


def synthetic_semantic_corpus():
    torch.manual_seed(3)
    return {
        "keys": torch.nn.functional.normalize(torch.randn(8, 6), dim=-1),
        "video_ids": ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"],
        "instance_ids": ["i{}".format(i) for i in range(8)],
        "coarse_class_ids": torch.tensor([2, 2, 3, 3, 4, 4, 1, 1]),
        "element_ids": torch.tensor([2, 2, 3, 3, 4, 4, 1, 1]),
        "elements": ["3A", "3A", "SP", "SP", "SQ", "SQ", "<background>", "<background>"],
        "goe_grades": torch.tensor([2.0, 1.0, -1.0, 0.0, 3.0, 2.0, 0.0, 0.0]),
        "bvs": torch.tensor([8.0, 8.0, 3.0, 3.0, 3.3, 3.3, 0.0, 0.0]),
        "panel_scores": torch.tensor([9.6, 8.8, 2.7, 3.0, 4.0, 3.7, 0.0, 0.0]),
        "valid_score_mask": torch.tensor([True, True, True, True, True, True, False, False]),
        "prototype_counts": torch.ones(8, dtype=torch.long),
        "coarse_class_vocab": ["unknown", "background", "jump", "spin", "sequence"],
        "element_vocab": ["<unknown>", "<background>", "3A", "SP", "SQ"],
        "metadata_version": "semantic-test-v1",
    }


class MultiPositiveLossTest(unittest.TestCase):
    def test_better_positive_logit_reduces_loss(self):
        positives = torch.tensor([[True, False, False]])
        weak, usable = multi_positive_nll(
            torch.tensor([[0.0, 1.0, 1.0]]), positives
        )
        strong, _ = multi_positive_nll(
            torch.tensor([[3.0, 1.0, 1.0]]), positives
        )
        self.assertTrue(bool(usable[0]))
        self.assertLess(float(strong), float(weak))

    def test_no_positive_row_is_ignored(self):
        logits = torch.randn(2, 3, requires_grad=True)
        loss, usable = multi_positive_nll(
            logits, torch.zeros(2, 3, dtype=torch.bool)
        )
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertFalse(usable.any())

    def test_soft_target_prefers_the_quality_matched_reference(self):
        target = torch.tensor([[0.9, 0.1, 0.0]])
        good, usable = soft_target_nll(torch.tensor([[3.0, 0.0, 0.0]]), target)
        bad, _ = soft_target_nll(torch.tensor([[0.0, 3.0, 0.0]]), target)
        self.assertTrue(bool(usable[0]))
        self.assertLess(float(good), float(bad))


class FineFSSemanticRAGTest(unittest.TestCase):
    def test_outputs_are_finite_and_goe_is_bounded(self):
        torch.manual_seed(4)
        model = FineFSSemanticRAG(
            coarse_classes=5,
            elements=5,
            query_dim=6,
            evidence_dim=8,
            encoder_hidden_dim=16,
            metadata_dim=4,
            dropout=0.0,
        )
        output = model(
            torch.randn(2, 6),
            torch.randn(2, 3, 6),
            torch.tensor([[2, 3, 1], [4, 2, 1]]),
            torch.tensor([[2, 3, 1], [4, 2, 1]]),
            torch.tensor([[2.0, -1.0, 0.0], [3.0, 1.0, 0.0]]),
            torch.ones(2, 3),
            torch.ones(2, 3),
            torch.tensor([[True, True, False], [True, True, False]]),
        )
        self.assertEqual(output["retrieval_logits"].shape, (2, 3))
        self.assertEqual(output["citation_weights"].shape, (2, 3))
        self.assertEqual(output["coarse_logits"].shape, (2, 5))
        self.assertEqual(output["element_logits"].shape, (2, 5))
        self.assertTrue(torch.isfinite(output["predicted_goe"]).all())
        self.assertTrue(torch.isfinite(output["direct_goe"]).all())
        self.assertTrue(output["predicted_goe"].abs().le(5.0).all())

    def test_invalid_candidate_receives_zero_weight(self):
        model = FineFSSemanticRAG(
            coarse_classes=5,
            elements=5,
            query_dim=6,
            evidence_dim=8,
            encoder_hidden_dim=16,
            metadata_dim=4,
            dropout=0.0,
        )
        output = model(
            torch.randn(1, 6),
            torch.randn(1, 2, 6),
            torch.tensor([[2, 3]]),
            torch.tensor([[2, 3]]),
            torch.tensor([[1.0, 0.0]]),
            torch.ones(1, 2),
            torch.ones(1, 2),
            torch.tensor([[True, True]]),
            candidate_valid_mask=torch.tensor([[True, False]]),
        )
        self.assertEqual(float(output["citation_weights"][0, 1]), 0.0)


class SemanticDataTest(unittest.TestCase):
    def test_minus_six_judge_sentinel_is_not_a_valid_goe(self):
        value = trimmed_judge_grade({"judge_score": [-6] * 9})
        self.assertTrue(torch.isnan(torch.tensor(value)))

    def test_video_split_is_deterministic_and_disjoint(self):
        corpus = synthetic_semantic_corpus()
        first = make_video_split(corpus, seed=11, train_ratio=0.5, val_ratio=0.25)
        second = make_video_split(corpus, seed=11, train_ratio=0.5, val_ratio=0.25)
        self.assertEqual(first, second)
        validate_video_split(corpus, first)

    def test_sampler_excludes_query_video_and_keeps_exact_positive(self):
        corpus = synthetic_semantic_corpus()
        sampler = SemanticCandidateSampler(
            corpus,
            train_video_ids=corpus["video_ids"],
            candidate_count=4,
            positive_count=1,
            seed=5,
        )
        selected = sampler.sample_one(0)
        self.assertTrue(all(corpus["video_ids"][i] != "v0" for i in selected))
        self.assertTrue(any(int(corpus["element_ids"][i]) == 2 for i in selected))

    def test_dataset_keeps_scored_actions_and_background(self):
        corpus = synthetic_semantic_corpus()
        dataset = FineFSSemanticDataset(corpus, ["v0", "v6"])
        self.assertEqual(set(dataset.indices), {0, 6})

    def test_device_corpus_gather_uses_numeric_video_ids(self):
        corpus = synthetic_semantic_corpus()
        device_corpus = prepare_device_corpus(corpus, torch.device("cpu"))
        batch = gather(device_corpus, torch.tensor([0, 2]))
        self.assertEqual(batch["features"].shape, (2, 6))
        self.assertEqual(batch["video"].dtype, torch.long)
        self.assertNotEqual(int(batch["video"][0]), int(batch["video"][1]))

    def test_retrieval_deduplication_stays_in_rank_order(self):
        top = torch.tensor([[0, 1, 2, 3, 4]])
        bank_indices = torch.tensor([10, 11, 12, 13, 14])
        bank_instances = torch.tensor([0, 0, 1, 2, 2])
        selected = deduplicated_indices(
            top, bank_indices, bank_instances, top_k=3
        )
        self.assertEqual(selected.tolist(), [[10, 12, 13]])


if __name__ == "__main__":
    unittest.main()
