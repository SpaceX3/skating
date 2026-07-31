import math
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


def masked_mean(values, mask, dim, keepdim=False):
    mask = mask.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    numerator = (values * mask).sum(dim=dim, keepdim=keepdim)
    denominator = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return numerator / denominator


def masked_softmax(logits, mask, dim=-1):
    mask = mask.bool()
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked_logits, dim=dim)
    weights = weights * mask.to(weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def load_action_corpus(path, map_location="cpu"):
    corpus = torch.load(path, map_location=map_location)
    required = {
        "keys",
        "video_ids",
        "instance_ids",
        "coarse_class_ids",
        "element_ids",
        "goe_grades",
        "bvs",
        "panel_scores",
        "valid_score_mask",
        "coarse_class_vocab",
        "element_vocab",
        "metadata_version",
    }
    missing = sorted(required.difference(corpus))
    if missing:
        raise KeyError("Action corpus is missing fields: {}".format(missing))
    keys = corpus["keys"].float()
    if keys.ndim != 2:
        raise ValueError("corpus keys must have shape [N,D]")
    if not torch.isfinite(keys).all():
        raise ValueError("corpus keys contain NaN/Inf")
    corpus["keys"] = F.normalize(keys, dim=-1)
    size = keys.shape[0]
    for name in (
        "video_ids",
        "instance_ids",
        "coarse_class_ids",
        "element_ids",
        "goe_grades",
        "bvs",
        "panel_scores",
        "valid_score_mask",
    ):
        if len(corpus[name]) != size:
            raise ValueError("corpus field {!r} has length {}, expected {}".format(
                name, len(corpus[name]), size
            ))
    ordered_fields = {
        "ordered_frame_features",
        "ordered_frame_times",
        "ordered_frame_mask",
    }
    present_ordered = ordered_fields.intersection(corpus)
    if present_ordered and present_ordered != ordered_fields:
        raise KeyError(
            "ordered action corpus must contain all of {}".format(
                sorted(ordered_fields)
            )
        )
    if present_ordered:
        frame_features = corpus["ordered_frame_features"]
        frame_times = corpus["ordered_frame_times"].float()
        frame_mask = corpus["ordered_frame_mask"].bool()
        if (
            frame_features.ndim != 3
            or frame_features.shape[0] != size
            or frame_features.shape[2] != keys.shape[1]
        ):
            raise ValueError(
                "ordered_frame_features must have shape [N,T,D] matching keys"
            )
        if frame_times.shape != frame_features.shape[:2]:
            raise ValueError("ordered_frame_times must have shape [N,T]")
        if frame_mask.shape != frame_features.shape[:2]:
            raise ValueError("ordered_frame_mask must have shape [N,T]")
        if not frame_mask.any(dim=1).all():
            raise ValueError("every ordered sequence must contain at least one frame")
        if not torch.isfinite(frame_features.float()).all():
            raise ValueError("ordered_frame_features contain NaN/Inf")
        if not torch.isfinite(frame_times[frame_mask]).all():
            raise ValueError("ordered_frame_times contain NaN/Inf on valid frames")
        corpus["ordered_frame_features"] = frame_features.half()
        corpus["ordered_frame_times"] = frame_times
        corpus["ordered_frame_mask"] = frame_mask
    return corpus


class EvidenceRAG(nn.Module):
    """Evidence-only score correction over fixed, precomputed retrieval results."""

    def __init__(
        self,
        corpus: Dict,
        dynamic_dim=512,
        query_dim=2048,
        evidence_dim=256,
        metadata_dim=32,
        delta_max=20.0,
    ):
        super().__init__()
        self.dynamic_dim = int(dynamic_dim)
        self.query_dim = int(query_dim)
        self.evidence_dim = int(evidence_dim)
        self.delta_max = float(delta_max)
        self.metadata_version = str(corpus["metadata_version"])
        self.video_ids = list(corpus["video_ids"])
        self.instance_ids = list(corpus["instance_ids"])
        self.elements = list(corpus.get("elements", [""] * len(self.video_ids)))

        self.register_buffer("corpus_keys", corpus["keys"].float(), persistent=False)
        self.register_buffer(
            "corpus_coarse_ids", corpus["coarse_class_ids"].long(), persistent=False
        )
        self.register_buffer(
            "corpus_element_ids", corpus["element_ids"].long(), persistent=False
        )
        self.register_buffer("corpus_goe", corpus["goe_grades"].float(), persistent=False)
        self.register_buffer("corpus_bv", corpus["bvs"].float(), persistent=False)
        self.register_buffer(
            "corpus_panel", corpus["panel_scores"].float(), persistent=False
        )
        self.register_buffer(
            "corpus_valid_score", corpus["valid_score_mask"].bool(), persistent=False
        )

        self.coarse_embedding = nn.Embedding(
            len(corpus["coarse_class_vocab"]), metadata_dim
        )
        self.element_embedding = nn.Embedding(
            len(corpus["element_vocab"]), metadata_dim
        )
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(query_dim + dynamic_dim),
            nn.Linear(query_dim + dynamic_dim, evidence_dim),
            nn.GELU(),
            nn.Linear(evidence_dim, evidence_dim),
        )
        reference_input_dim = query_dim + 2 * metadata_dim + 4
        self.reference_encoder = nn.Sequential(
            nn.LayerNorm(reference_input_dim),
            nn.Linear(reference_input_dim, evidence_dim),
            nn.GELU(),
            nn.Linear(evidence_dim, evidence_dim),
        )
        pair_dim = 2 * evidence_dim + 4
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, evidence_dim),
            nn.GELU(),
            nn.Linear(evidence_dim, evidence_dim),
        )
        self.citation_head = nn.Linear(evidence_dim, 1)
        self.relative_goe_head = nn.Sequential(
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, evidence_dim // 2),
            nn.GELU(),
            nn.Linear(evidence_dim // 2, 1),
        )
        self.local_evidence_head = nn.Sequential(
            nn.LayerNorm(evidence_dim + 2),
            nn.Linear(evidence_dim + 2, evidence_dim),
            nn.GELU(),
            nn.Linear(evidence_dim, evidence_dim),
        )

        statistics_dim = 9
        correction_input_dim = 2 * evidence_dim + statistics_dim
        self.correction_head = nn.Sequential(
            nn.LayerNorm(correction_input_dim),
            nn.Linear(correction_input_dim, evidence_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(evidence_dim, 1),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def _gather(self, values, safe_indices):
        return values[safe_indices]

    def forward(
        self,
        dynamic_time_feat,
        static_raw,
        dynamic_valid_mask,
        static_valid_mask,
        candidate_indices,
        candidate_similarities,
        overlap_weights,
    ):
        if candidate_indices.ndim != 3:
            raise ValueError("candidate_indices must have shape [B,T,K]")
        if candidate_indices.shape != candidate_similarities.shape:
            raise ValueError("candidate index/similarity shapes must match")
        if static_raw.shape[:2] != candidate_indices.shape[:2]:
            raise ValueError("static features and candidates must share [B,T]")
        if overlap_weights.shape != (
            static_raw.shape[0],
            static_raw.shape[1],
            dynamic_time_feat.shape[1],
        ):
            raise ValueError("overlap_weights must have shape [B,T_static,T_dynamic]")

        candidate_valid = candidate_indices.ge(0)
        candidate_valid = candidate_valid & static_valid_mask.unsqueeze(-1)
        if candidate_valid.any() and candidate_indices[candidate_valid].max() >= len(
            self.corpus_keys
        ):
            raise IndexError("candidate index is outside the loaded action corpus")
        safe_indices = candidate_indices.clamp_min(0)

        reference_keys = self._gather(self.corpus_keys, safe_indices)
        coarse_ids = self._gather(self.corpus_coarse_ids, safe_indices)
        element_ids = self._gather(self.corpus_element_ids, safe_indices)
        reference_goe = self._gather(self.corpus_goe, safe_indices)
        reference_bv = self._gather(self.corpus_bv, safe_indices)
        reference_panel = self._gather(self.corpus_panel, safe_indices)
        score_reference_mask = candidate_valid & self._gather(
            self.corpus_valid_score, safe_indices
        )

        valid_overlap = overlap_weights.to(dynamic_time_feat.dtype)
        valid_overlap = valid_overlap * dynamic_valid_mask.unsqueeze(1).to(
            dynamic_time_feat.dtype
        )
        valid_overlap = valid_overlap * static_valid_mask.unsqueeze(-1).to(
            dynamic_time_feat.dtype
        )
        dynamic_context = torch.bmm(valid_overlap, dynamic_time_feat)
        dynamic_context = dynamic_context / valid_overlap.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        query_token = self.query_encoder(
            torch.cat([static_raw, dynamic_context], dim=-1)
        )

        coarse_token = self.coarse_embedding(coarse_ids)
        element_token = self.element_embedding(element_ids)
        numeric_reference = torch.stack(
            [
                reference_goe / 5.0,
                reference_bv / 15.0,
                reference_panel / 20.0,
                candidate_similarities.clamp(-1.0, 1.0),
            ],
            dim=-1,
        )
        reference_token = self.reference_encoder(
            torch.cat(
                [reference_keys, coarse_token, element_token, numeric_reference], dim=-1
            )
        )
        expanded_query = query_token.unsqueeze(2).expand_as(reference_token)
        pair_features = torch.cat(
            [
                torch.abs(expanded_query - reference_token),
                expanded_query * reference_token,
                numeric_reference,
            ],
            dim=-1,
        )
        pair_token = self.pair_encoder(pair_features)

        citation_logits = self.citation_head(pair_token).squeeze(-1)
        citation_weights = masked_softmax(citation_logits, candidate_valid, dim=-1)
        score_weights = citation_weights * score_reference_mask.to(citation_weights.dtype)
        retrieval_confidence = score_weights.sum(dim=-1)

        delta_goe = 5.0 * torch.tanh(self.relative_goe_head(pair_token).squeeze(-1))
        candidate_goe = reference_goe + delta_goe
        pred_goe = (score_weights * candidate_goe).sum(dim=-1)
        pred_goe = pred_goe / retrieval_confidence.clamp_min(1e-12)
        pred_goe = torch.where(
            retrieval_confidence.gt(0), pred_goe, torch.zeros_like(pred_goe)
        )

        local_pair_evidence = self.local_evidence_head(
            torch.cat(
                [
                    pair_token,
                    delta_goe.unsqueeze(-1) / 5.0,
                    candidate_similarities.unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        local_evidence = (
            score_weights.unsqueeze(-1) * local_pair_evidence
        ).sum(dim=2)
        static_mask_float = static_valid_mask.to(local_evidence.dtype)
        evidence_mean = masked_mean(local_evidence, static_valid_mask, dim=1)

        max_input = local_evidence.masked_fill(
            ~static_valid_mask.unsqueeze(-1), torch.finfo(local_evidence.dtype).min
        )
        evidence_max = max_input.max(dim=1).values
        any_static = static_valid_mask.any(dim=1, keepdim=True)
        evidence_max = torch.where(any_static, evidence_max, torch.zeros_like(evidence_max))

        valid_candidate_sims = candidate_similarities.masked_fill(
            ~candidate_valid, torch.finfo(candidate_similarities.dtype).min
        )
        sorted_sims = valid_candidate_sims.sort(dim=-1, descending=True).values
        top1 = sorted_sims[..., 0]
        if sorted_sims.shape[-1] > 1:
            top2 = sorted_sims[..., 1]
        else:
            top2 = torch.zeros_like(top1)
        valid_count = candidate_valid.sum(dim=-1)
        top1 = torch.where(valid_count.ge(1), top1, torch.zeros_like(top1))
        top2 = torch.where(valid_count.ge(2), top2, torch.zeros_like(top2))
        margin = top1 - top2
        entropy = -(
            citation_weights.clamp_min(1e-12).log() * citation_weights
        ).sum(dim=-1)
        entropy = entropy / max(math.log(max(candidate_indices.shape[-1], 2)), 1e-6)

        confidence_mean = masked_mean(
            retrieval_confidence.unsqueeze(-1), static_valid_mask, dim=1
        ).squeeze(-1)
        confidence_max = retrieval_confidence.masked_fill(
            ~static_valid_mask, 0.0
        ).max(dim=1).values
        top1_mean = masked_mean(top1.unsqueeze(-1), static_valid_mask, dim=1).squeeze(-1)
        top1_max = top1.masked_fill(~static_valid_mask, 0.0).max(dim=1).values
        margin_mean = masked_mean(
            margin.unsqueeze(-1), static_valid_mask, dim=1
        ).squeeze(-1)
        entropy_mean = masked_mean(
            entropy.unsqueeze(-1), static_valid_mask, dim=1
        ).squeeze(-1)

        goe_mask = static_valid_mask & retrieval_confidence.gt(0)
        goe_mean = masked_mean(pred_goe.unsqueeze(-1), goe_mask, dim=1).squeeze(-1)
        goe_variance = masked_mean(
            (pred_goe - goe_mean.unsqueeze(-1)).pow(2).unsqueeze(-1),
            goe_mask,
            dim=1,
        ).squeeze(-1)
        goe_std = torch.sqrt(goe_variance.clamp_min(0.0))
        valid_window_ratio = (
            (goe_mask.to(static_mask_float.dtype).sum(dim=1))
            / static_mask_float.sum(dim=1).clamp_min(1.0)
        )
        statistics = torch.stack(
            [
                confidence_mean,
                confidence_max,
                top1_mean,
                top1_max,
                margin_mean,
                entropy_mean,
                goe_mean / 5.0,
                goe_std / 5.0,
                valid_window_ratio,
            ],
            dim=-1,
        )

        correction_input = torch.cat(
            [evidence_mean, evidence_max, statistics], dim=-1
        )
        raw_delta = self.correction_head(correction_input).squeeze(-1)
        evidence_present = goe_mask.any(dim=1)
        delta_tes = self.delta_max * torch.tanh(raw_delta)
        delta_tes = delta_tes * evidence_present.to(delta_tes.dtype)

        return {
            "delta_tes_rag": delta_tes,
            "pred_goe_grade": pred_goe,
            "citation_indices": candidate_indices,
            "citation_weights": citation_weights,
            "citation_logits": citation_logits,
            "retrieval_confidence": retrieval_confidence,
            "retrieval_statistics": statistics,
            "evidence_valid_mask": evidence_present,
        }

    def resolve_citations(self, indices):
        resolved = []
        for index in indices:
            index = int(index)
            if index < 0:
                resolved.append(None)
                continue
            resolved.append(
                {
                    "corpus_index": index,
                    "video_id": self.video_ids[index],
                    "instance_id": self.instance_ids[index],
                    "element": self.elements[index],
                    "reference_goe": float(self.corpus_goe[index].detach().cpu()),
                    "reference_bv": float(self.corpus_bv[index].detach().cpu()),
                }
            )
        return resolved


class SemanticEvidenceRAG(nn.Module):
    """Frozen FineFS semantic evidence followed by a trainable TES residual."""

    def __init__(self, corpus, semantic_model, dynamic_dim=512, delta_max=20.0):
        super().__init__()
        self.semantic = semantic_model
        self.dynamic_dim = int(dynamic_dim)
        self.evidence_dim = int(semantic_model.evidence_dim)
        self.metadata_dim = int(semantic_model.goe_model.element_embedding.embedding_dim)
        self.delta_max = float(delta_max)
        self.metadata_version = str(corpus["metadata_version"])
        self.video_ids = list(corpus["video_ids"])
        self.instance_ids = list(corpus["instance_ids"])
        self.elements = list(corpus.get("elements", [""] * len(self.video_ids)))

        self.register_buffer("corpus_keys", corpus["keys"].float(), persistent=False)
        self.register_buffer("corpus_coarse_ids", corpus["coarse_class_ids"].long(), persistent=False)
        self.register_buffer("corpus_element_ids", corpus["element_ids"].long(), persistent=False)
        self.register_buffer("corpus_goe", corpus["goe_grades"].float(), persistent=False)
        self.register_buffer("corpus_bv", corpus["bvs"].float(), persistent=False)
        self.register_buffer("corpus_panel", corpus["panel_scores"].float(), persistent=False)
        self.register_buffer("corpus_valid_score", corpus["valid_score_mask"].bool(), persistent=False)

        for parameter in self.semantic.parameters():
            parameter.requires_grad = False
        self.semantic.eval()

        self.dynamic_projection = nn.Sequential(
            nn.LayerNorm(self.dynamic_dim),
            nn.Linear(self.dynamic_dim, self.evidence_dim),
            nn.GELU(),
        )
        self.window_encoder = nn.Sequential(
            nn.LayerNorm(2 * self.evidence_dim + 2 * self.metadata_dim + 14),
            nn.Linear(2 * self.evidence_dim + 2 * self.metadata_dim + 14, self.evidence_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.evidence_dim, self.evidence_dim),
        )
        self.correction_head = nn.Sequential(
            nn.LayerNorm(2 * self.evidence_dim + 15),
            nn.Linear(2 * self.evidence_dim + 15, self.evidence_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.evidence_dim, 1),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.semantic.eval()
        return self

    def forward(
        self,
        dynamic_time_feat,
        static_raw,
        dynamic_valid_mask,
        static_valid_mask,
        candidate_indices,
        candidate_similarities,
        overlap_weights,
        tes_baseline,
    ):
        if candidate_indices.ndim != 3 or candidate_indices.shape != candidate_similarities.shape:
            raise ValueError("semantic candidates must share shape [B,T,K]")
        if static_raw.shape[:2] != candidate_indices.shape[:2]:
            raise ValueError("static features and semantic candidates must share [B,T]")
        batch_size, static_steps, candidate_count = candidate_indices.shape
        candidate_valid = candidate_indices.ge(0) & static_valid_mask.unsqueeze(-1)
        if candidate_valid.any() and candidate_indices[candidate_valid].max() >= len(self.corpus_keys):
            raise IndexError("candidate index is outside the loaded action corpus")
        safe_indices = candidate_indices.clamp_min(0)

        overlap = overlap_weights.to(dynamic_time_feat.dtype)
        overlap = overlap * dynamic_valid_mask.unsqueeze(1).to(overlap.dtype)
        overlap = overlap * static_valid_mask.unsqueeze(-1).to(overlap.dtype)
        dynamic_context = torch.bmm(overlap, dynamic_time_feat)
        dynamic_context = dynamic_context / overlap.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        reference_features = self.corpus_keys[safe_indices]
        reference_coarse = self.corpus_coarse_ids[safe_indices]
        reference_elements = self.corpus_element_ids[safe_indices]
        reference_goe = self.corpus_goe[safe_indices]
        reference_bv = self.corpus_bv[safe_indices]
        reference_panel = self.corpus_panel[safe_indices]
        reference_valid_score = self.corpus_valid_score[safe_indices] & candidate_valid
        semantic_query = F.normalize(static_raw, dim=-1)
        raw_cosine = F.cosine_similarity(semantic_query.unsqueeze(2), reference_features, dim=-1)

        flat = lambda value: value.reshape(batch_size * static_steps, *value.shape[2:])
        with torch.no_grad():
            semantic = self.semantic(
                query_features=semantic_query.reshape(batch_size * static_steps, -1),
                reference_features=flat(reference_features),
                reference_coarse_ids=flat(reference_coarse),
                reference_element_ids=flat(reference_elements),
                reference_goe=flat(reference_goe),
                reference_bv=flat(reference_bv),
                reference_panel=flat(reference_panel),
                reference_score_valid=flat(reference_valid_score),
                candidate_valid_mask=flat(candidate_valid),
                candidate_similarities=flat(raw_cosine),
            )

        def windows(name):
            return semantic[name].reshape(batch_size, static_steps, *semantic[name].shape[1:])

        query_token = windows("query_token")
        citation_weights = windows("citation_weights")
        element_probabilities = windows("element_probabilities")
        coarse_probabilities = windows("coarse_probabilities")
        expected_element = element_probabilities @ self.semantic.goe_model.element_embedding.weight
        expected_coarse = coarse_probabilities @ self.semantic.goe_model.coarse_embedding.weight
        element_entropy = -(element_probabilities.clamp_min(1e-12).log() * element_probabilities).sum(-1)
        element_entropy = element_entropy / max(math.log(max(element_probabilities.shape[-1], 2)), 1e-6)
        coarse_entropy = -(coarse_probabilities.clamp_min(1e-12).log() * coarse_probabilities).sum(-1)
        coarse_entropy = coarse_entropy / max(math.log(max(coarse_probabilities.shape[-1], 2)), 1e-6)
        citation_entropy = -(citation_weights.clamp_min(1e-12).log() * citation_weights).sum(-1)
        citation_entropy = citation_entropy / max(math.log(max(candidate_count, 2)), 1e-6)

        valid_scores = candidate_similarities.masked_fill(~candidate_valid, torch.finfo(candidate_similarities.dtype).min)
        sorted_scores = valid_scores.sort(dim=-1, descending=True).values
        valid_count = candidate_valid.sum(-1)
        top1 = torch.where(valid_count.ge(1), sorted_scores[..., 0], torch.zeros_like(sorted_scores[..., 0]))
        top2 = sorted_scores[..., 1] if candidate_count > 1 else torch.zeros_like(top1)
        top2 = torch.where(valid_count.ge(2), top2, torch.zeros_like(top2))
        margin = top1 - top2

        direct_goe = windows("direct_goe")
        evidence_reference_goe = windows("evidence_reference_goe")
        evidence_delta_goe = windows("evidence_delta_goe")
        evidence_goe = windows("evidence_goe")
        predicted_goe = windows("predicted_goe")
        goe_gate = windows("goe_gate")
        goe_confidence = windows("goe_confidence").clamp(0.0, 1.0)
        valid_ratio = valid_count.to(static_raw.dtype) / max(candidate_count, 1)
        scalar_features = torch.stack(
            [
                direct_goe / 5.0,
                evidence_reference_goe / 5.0,
                evidence_delta_goe / 5.0,
                evidence_goe / 5.0,
                predicted_goe / 5.0,
                goe_gate,
                goe_confidence,
                windows("action_probability"),
                element_entropy,
                coarse_entropy,
                citation_entropy,
                torch.tanh(top1 / 10.0),
                torch.tanh(margin / 10.0),
                valid_ratio,
            ],
            dim=-1,
        )
        window_token = self.window_encoder(
            torch.cat([
                query_token, expected_element, expected_coarse,
                self.dynamic_projection(dynamic_context), scalar_features
            ], dim=-1)
        )
        window_mask = static_valid_mask
        evidence_mean = masked_mean(window_token, window_mask, dim=1)
        evidence_max = window_token.masked_fill(
            ~window_mask.unsqueeze(-1), torch.finfo(window_token.dtype).min
        ).max(dim=1).values
        evidence_max = torch.where(window_mask.any(1, keepdim=True), evidence_max, torch.zeros_like(evidence_max))
        semantic_statistics = masked_mean(scalar_features, window_mask, dim=1)
        correction_input = torch.cat(
            [evidence_mean, evidence_max, semantic_statistics, tes_baseline.unsqueeze(-1) / 100.0], dim=-1
        )
        raw_delta = self.correction_head(correction_input).squeeze(-1)
        video_confidence = semantic_statistics[:, 6].clamp(0.0, 1.0)
        confidence_gate = 0.25 + 0.75 * video_confidence
        evidence_present = window_mask.any(dim=1)
        delta_tes = self.delta_max * torch.tanh(raw_delta) * confidence_gate
        delta_tes = delta_tes * evidence_present.to(delta_tes.dtype)

        return {
            "delta_tes_rag": delta_tes,
            "pred_goe_grade": predicted_goe,
            "direct_goe_grade": direct_goe,
            "evidence_goe_grade": evidence_goe,
            "evidence_reference_goe_grade": evidence_reference_goe,
            "evidence_delta_goe_grade": evidence_delta_goe,
            "expected_element_embedding": expected_element,
            "expected_coarse_embedding": expected_coarse,
            "semantic_goe_gate": goe_gate,
            "citation_indices": candidate_indices,
            "citation_weights": citation_weights,
            "retrieval_confidence": goe_confidence,
            "retrieval_statistics": semantic_statistics,
            "evidence_valid_mask": evidence_present,
        }

    def resolve_citations(self, indices):
        resolved = []
        for index in indices:
            index = int(index)
            if index < 0:
                resolved.append(None)
                continue
            resolved.append(
                {
                    "corpus_index": index,
                    "video_id": self.video_ids[index],
                    "instance_id": self.instance_ids[index],
                    "element": self.elements[index],
                    "reference_goe": float(self.corpus_goe[index].detach().cpu()),
                    "reference_bv": float(self.corpus_bv[index].detach().cpu()),
                }
            )
        return resolved
