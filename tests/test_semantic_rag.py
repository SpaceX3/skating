import unittest

import torch

from semantic_data import (
    FineFSSemanticDataset,
    SemanticCandidateSampler,
    make_video_split,
    validate_video_split,
)
from semantic_main import deduplicated_indices, gather, prepare_device_corpus
from scripts.build_action_rag_corpus import pack_ordered_sequence, trimmed_judge_grade
from semantic_rag import (
    FineFSSemanticRAG,
    OrderedInteractionEncoder,
    SemanticQueryClassifier,
    load_classifier_state,
    require_candidate_v2,
    require_semantic_v2,
    multi_positive_nll,
    soft_target_nll,
)


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
    def make_model(self):
        return FineFSSemanticRAG(5, 5, 6, 8, 16, 4, dropout=0.0)

    def test_classifier_inference_has_no_label_argument(self):
        classifier = SemanticQueryClassifier(5, 5, 6, 8, 16, 0.0)
        output = classifier.classify_query(torch.randn(2, 6))
        self.assertEqual(output["element_probabilities"].shape, (2, 5))
        with self.assertRaises(TypeError):
            classifier.classify_query(torch.randn(2, 6), torch.tensor([1, 2]))

    def test_stage_a_loss_trains_every_classifier_parameter(self):
        """No classifier parameter may be frozen at random initialisation.

        The retrieval projection used to live here and never received Stage A
        gradients, so it entered Stage B random and frozen.
        """
        classifier = SemanticQueryClassifier(5, 5, 6, 8, 16, 0.0)
        output = classifier(torch.randn(4, 6))
        loss = (
            output["action_logit"].square().mean()
            + output["coarse_logits"].square().mean()
            + output["element_logits"].square().mean()
        )
        loss.backward()
        missing = [
            name
            for name, parameter in classifier.named_parameters()
            if parameter.grad is None
        ]
        self.assertEqual(missing, [])

    def test_ordered_interaction_reverses_antisymmetric_pairs(self):
        torch.manual_seed(17)
        encoder = OrderedInteractionEncoder(
            input_dim=6,
            output_dim=8,
            projection_dim=3,
            time_basis=1,
            hidden_dim=8,
            dropout=0.0,
        )
        frames = torch.randn(1, 4, 6)
        times = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        _, forward_pairs = encoder.path_features(frames, times, mask)
        _, reverse_pairs = encoder.path_features(frames.flip(1), times, mask)
        self.assertGreater(float(forward_pairs.abs().sum()), 0.0)
        self.assertTrue(
            torch.allclose(reverse_pairs, -forward_pairs, atol=1e-5, rtol=1e-5)
        )

    def test_ordered_classifier_uses_sequence_without_mean_encoder(self):
        classifier = SemanticQueryClassifier(
            5,
            5,
            6,
            8,
            16,
            0.0,
            ordered_enabled=True,
            ordered_projection_dim=3,
            ordered_time_basis=2,
            ordered_hidden_dim=12,
        )
        self.assertIsNone(classifier.query_encoder)
        with self.assertRaisesRegex(ValueError, "requires frame features"):
            classifier(torch.randn(2, 6))
        frame_features = torch.randn(2, 4, 6)
        frame_times = torch.arange(4).repeat(2, 1).float()
        frame_mask = torch.ones(2, 4, dtype=torch.bool)
        output = classifier(
            torch.randn(2, 6),
            ordered_frame_features=frame_features,
            ordered_frame_times=frame_times,
            ordered_frame_mask=frame_mask,
        )
        changed_mean_output = classifier(
            torch.randn(2, 6) * 100.0,
            ordered_frame_features=frame_features,
            ordered_frame_times=frame_times,
            ordered_frame_mask=frame_mask,
        )
        self.assertEqual(output["query_token"].shape, (2, 8))
        for name in ("query_token", "action_logit", "coarse_logits", "element_logits"):
            self.assertTrue(torch.equal(output[name], changed_mean_output[name]))
        loss = (
            output["action_logit"].square().mean()
            + output["coarse_logits"].square().mean()
            + output["element_logits"].square().mean()
        )
        loss.backward()
        missing = [
            name
            for name, parameter in classifier.named_parameters()
            if parameter.grad is None
        ]
        self.assertEqual(missing, [])

    def test_query_retrieval_projection_is_trained_by_stage_b(self):
        """Soft routing must use a projection owned by the trainable GOE stage."""
        model = self.make_model()
        model.freeze_classifier()
        trainable = dict(model.goe_model.named_parameters())
        self.assertIn("query_retrieval_encoder.weight", trainable)
        output = model.classify_query(torch.randn(2, 6))
        output["query_retrieval"].square().mean().backward()
        self.assertIsNotNone(trainable["query_retrieval_encoder.weight"].grad)
        for parameter in model.classifier.parameters():
            self.assertIsNone(parameter.grad)

    def test_retrieval_logits_reach_the_query_projection(self):
        model = self.make_model()
        model.freeze_classifier()
        output = model(
            torch.randn(1, 6), torch.randn(1, 2, 6), torch.tensor([[2, 3]]),
            torch.tensor([[2, 3]]), torch.tensor([[1.0, -1.0]]),
            torch.ones(1, 2), torch.ones(1, 2), torch.tensor([[True, True]]),
        )
        output["retrieval_logits"].sum().backward()
        gradient = model.goe_model.query_retrieval_encoder.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_stale_classifier_retrieval_keys_are_dropped(self):
        classifier = SemanticQueryClassifier(5, 5, 6, 8, 16, 0.0)
        state = dict(classifier.state_dict())
        state["retrieval_encoder.weight"] = torch.randn(8, 8)
        state["retrieval_encoder.bias"] = torch.randn(8)
        retired = load_classifier_state(classifier, state)
        self.assertEqual(
            retired, ["retrieval_encoder.bias", "retrieval_encoder.weight"]
        )

    def test_soft_route_probability_is_monotonic_and_not_hard_top1(self):
        model = self.make_model()
        query_retrieval = torch.tensor([[1.0, 0.0, 0, 0, 0, 0, 0, 0]])
        references = torch.tensor([[1.0, 0.0, 0, 0, 0, 0, 0, 0], [0.9, 0.1, 0, 0, 0, 0, 0, 0]])
        references = torch.nn.functional.normalize(references, dim=-1)
        coarse = torch.tensor([2, 2]); elements = torch.tensor([2, 3])
        coarse_p = torch.tensor([[0.01, 0.01, 0.96, 0.01, 0.01]])
        low = torch.tensor([[0.01, 0.01, 0.10, 0.86, 0.02]])
        high = torch.tensor([[0.01, 0.01, 0.40, 0.56, 0.02]])
        score_low = model.retrieval_scores(query_retrieval, torch.tensor([0.9]), coarse_p, low, references, coarse, elements)
        score_high = model.retrieval_scores(query_retrieval, torch.tensor([0.9]), coarse_p, high, references, coarse, elements)
        self.assertGreater(float(score_high[0, 0]), float(score_low[0, 0]))
        self.assertTrue(torch.isfinite(score_low[0, 0]))  # element 2 is not top-1

    def test_full_bank_and_per_query_scores_match(self):
        model = self.make_model(); torch.manual_seed(8)
        q = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
        r = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
        action = torch.tensor([0.8, 0.3]); coarse_p = torch.softmax(torch.randn(2, 5), -1); element_p = torch.softmax(torch.randn(2, 5), -1)
        coarse = torch.tensor([1, 2, 3]); element = torch.tensor([1, 2, 3])
        full = model.retrieval_scores(q, action, coarse_p, element_p, r, coarse, element)
        per = model.retrieval_scores(q, action, coarse_p, element_p, r.unsqueeze(0).expand(2, -1, -1), coarse.unsqueeze(0).expand(2, -1), element.unsqueeze(0).expand(2, -1))
        self.assertTrue(torch.allclose(full, per, atol=1e-6))

    def test_goe_decomposition_invalid_mask_and_direct_fallback(self):
        model = self.make_model()
        output = model(torch.randn(1, 6), torch.randn(1, 2, 6), torch.tensor([[2, 3]]), torch.tensor([[2, 3]]), torch.tensor([[1.0, -1.0]]), torch.ones(1, 2), torch.ones(1, 2), torch.tensor([[False, False]]))
        self.assertTrue(torch.equal(output["goe_evidence_weights"], torch.zeros_like(output["goe_evidence_weights"])))
        self.assertTrue(torch.allclose(output["predicted_goe"], output["direct_goe"]))
        self.assertTrue(torch.isfinite(output["predicted_goe"]).all())
        self.assertTrue(torch.allclose(output["evidence_goe_unbounded"], output["evidence_reference_goe"] + output["evidence_delta_goe"]))

    def test_v1_checkpoint_rejected_clearly(self):
        with self.assertRaisesRegex(ValueError, "format_version.*v1 checkpoints"):
            require_semantic_v2({"training_stage": "finefs_semantic"})
        with self.assertRaisesRegex(ValueError, "v1 candidates"):
            require_candidate_v2({"candidate_indices": []})

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
    def test_ordered_sequence_pack_is_chronological_and_masked(self):
        features = torch.arange(2 * 3 * 2).reshape(2, 3, 2).numpy()
        times = torch.tensor([[2.0, 1.0, 3.0], [6.0, 4.0, 5.0]]).numpy()
        packed, packed_times, mask = pack_ordered_sequence(
            features, times, torch.tensor([0, 1]).numpy(), 8
        )
        self.assertEqual(mask.sum(), 6)
        self.assertTrue((packed_times[:6] == sorted(packed_times[:6])).all())
        self.assertEqual(packed.shape, (8, 2))

    def test_ordered_corpus_moves_only_gathered_frames_to_device(self):
        corpus = synthetic_semantic_corpus()
        corpus["ordered_frame_features"] = torch.randn(8, 4, 6).half()
        corpus["ordered_frame_times"] = torch.arange(4).repeat(8, 1).float()
        corpus["ordered_frame_mask"] = torch.ones(8, 4, dtype=torch.bool)
        prepared = prepare_device_corpus(corpus, torch.device("cpu"))
        batch = gather(prepared, torch.tensor([1, 3]))
        self.assertEqual(batch["ordered_frame_features"].shape, (2, 4, 6))
        self.assertEqual(batch["ordered_frame_features"].dtype, torch.float32)

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
