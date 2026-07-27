import math

import torch
from torch import nn
import torch.nn.functional as F

from action_rag import masked_softmax


def multi_positive_nll(logits, positive_mask, valid_mask=None, sample_weights=None):
    """Listwise NLL where any positive candidate is an acceptable citation."""
    positive_mask = positive_mask.bool()
    if valid_mask is None:
        valid_mask = torch.ones_like(positive_mask)
    else:
        valid_mask = valid_mask.bool()
    positive_mask = positive_mask & valid_mask
    usable = positive_mask.any(dim=-1) & valid_mask.any(dim=-1)
    if not usable.any():
        return logits.sum() * 0.0, usable

    floor = torch.finfo(logits.dtype).min
    denominator = torch.logsumexp(logits.masked_fill(~valid_mask, floor), dim=-1)
    numerator = torch.logsumexp(logits.masked_fill(~positive_mask, floor), dim=-1)
    losses = denominator - numerator
    if sample_weights is None:
        return losses[usable].mean(), usable
    weights = sample_weights.to(losses.dtype)[usable]
    return (losses[usable] * weights).sum() / weights.sum().clamp_min(1e-12), usable


def soft_target_nll(logits, target_weights, valid_mask=None, sample_weights=None):
    """Listwise cross entropy with non-negative, row-normalized soft targets."""
    if valid_mask is None:
        valid_mask = torch.ones_like(target_weights, dtype=torch.bool)
    else:
        valid_mask = valid_mask.bool()
    target_weights = target_weights.to(logits.dtype) * valid_mask.to(logits.dtype)
    target_sum = target_weights.sum(dim=-1)
    usable = target_sum.gt(0) & valid_mask.any(dim=-1)
    if not usable.any():
        return logits.sum() * 0.0, usable
    targets = target_weights / target_sum.unsqueeze(-1).clamp_min(1e-12)
    floor = torch.finfo(logits.dtype).min
    log_probabilities = F.log_softmax(logits.masked_fill(~valid_mask, floor), dim=-1)
    losses = -(targets * log_probabilities).sum(dim=-1)
    if sample_weights is None:
        return losses[usable].mean(), usable
    weights = sample_weights.to(losses.dtype)[usable]
    return (losses[usable] * weights).sum() / weights.sum().clamp_min(1e-12), usable


class FineFSSemanticRAG(nn.Module):
    """Action retrieval, citation and GOE supervision before TES training."""

    def __init__(
        self,
        coarse_classes,
        elements,
        query_dim=2048,
        evidence_dim=256,
        encoder_hidden_dim=512,
        metadata_dim=64,
        temperature=0.07,
        dropout=0.1,
    ):
        super().__init__()
        self.query_dim = int(query_dim)
        self.evidence_dim = int(evidence_dim)
        self.coarse_classes = int(coarse_classes)
        self.elements = int(elements)

        self.coarse_embedding = nn.Embedding(self.coarse_classes, metadata_dim)
        self.element_embedding = nn.Embedding(self.elements, metadata_dim)
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, encoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_dim, evidence_dim),
        )
        # Query-dependent similarity is intentionally excluded so the reference
        # bank can be encoded once and indexed for retrieval.
        reference_input_dim = query_dim + 2 * metadata_dim + 3
        self.reference_encoder = nn.Sequential(
            nn.LayerNorm(reference_input_dim),
            nn.Linear(reference_input_dim, encoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_dim, evidence_dim),
        )
        pair_dim = 2 * evidence_dim + 4
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, encoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_dim, evidence_dim),
        )
        self.citation_head = nn.Linear(evidence_dim, 1)
        self.relative_goe_head = nn.Sequential(
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, evidence_dim // 2),
            nn.GELU(),
            nn.Linear(evidence_dim // 2, 1),
        )
        self.action_head = nn.Linear(evidence_dim, 1)
        self.coarse_head = nn.Linear(evidence_dim, self.coarse_classes)
        self.element_head = nn.Linear(evidence_dim, self.elements)
        self.direct_goe_head = nn.Sequential(
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, evidence_dim // 2),
            nn.GELU(),
            nn.Linear(evidence_dim // 2, 1),
        )
        self.goe_gate_head = nn.Sequential(
            nn.LayerNorm(evidence_dim),
            nn.Linear(evidence_dim, 1),
        )
        nn.init.constant_(self.goe_gate_head[-1].bias, -1.0)
        self.register_buffer(
            "element_goe_prior", torch.zeros(self.elements, dtype=torch.float32)
        )
        self.element_retrieval_weight = nn.Parameter(torch.tensor(0.0))
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / float(temperature)), dtype=torch.float32)
        )

    def set_element_goe_prior(self, values):
        values = torch.as_tensor(
            values,
            dtype=self.element_goe_prior.dtype,
            device=self.element_goe_prior.device,
        )
        if values.shape != self.element_goe_prior.shape:
            raise ValueError("element GOE prior has the wrong shape")
        self.element_goe_prior.copy_(values)

    def encode_query(self, query_features):
        if query_features.ndim != 2 or query_features.shape[-1] != self.query_dim:
            raise ValueError("query_features must have shape [B,query_dim]")
        token = self.query_encoder(query_features)
        return token, F.normalize(token, dim=-1)

    def retrieval_scores(
        self,
        query_retrieval,
        element_logits,
        reference_retrieval,
        reference_element_ids,
    ):
        scale = self.logit_scale.exp().clamp(max=100.0)
        if reference_retrieval.ndim == 2:
            visual_scores = query_retrieval @ reference_retrieval.T
            element_scores = F.log_softmax(element_logits, dim=-1)[
                :, reference_element_ids
            ]
        elif reference_retrieval.ndim == 3:
            visual_scores = (
                query_retrieval.unsqueeze(1) * reference_retrieval
            ).sum(dim=-1)
            element_scores = F.log_softmax(element_logits, dim=-1).gather(
                1, reference_element_ids
            )
        else:
            raise ValueError("reference retrieval embeddings must be 2D or 3D")
        element_weight = F.softplus(self.element_retrieval_weight)
        return scale * visual_scores + element_weight * element_scores

    def reference_numeric(self, goe, bv, panel):
        return torch.stack(
            [
                goe / 5.0,
                bv / 15.0,
                panel / 20.0,
            ],
            dim=-1,
        )

    def encode_reference(
        self,
        reference_features,
        coarse_ids,
        element_ids,
        goe,
        bv,
        panel,
    ):
        numeric = self.reference_numeric(goe, bv, panel)
        token = self.reference_encoder(
            torch.cat(
                [
                    reference_features,
                    self.coarse_embedding(coarse_ids),
                    self.element_embedding(element_ids),
                    numeric,
                ],
                dim=-1,
            )
        )
        return token, F.normalize(token, dim=-1), numeric

    def forward(
        self,
        query_features,
        reference_features,
        reference_coarse_ids,
        reference_element_ids,
        reference_goe,
        reference_bv,
        reference_panel,
        reference_score_valid,
        candidate_valid_mask=None,
        candidate_similarities=None,
    ):
        if reference_features.ndim != 3:
            raise ValueError("reference_features must have shape [B,K,D]")
        batch_size, candidate_count = reference_features.shape[:2]
        if query_features.shape[0] != batch_size:
            raise ValueError("query/reference batch sizes differ")
        if candidate_valid_mask is None:
            candidate_valid_mask = torch.ones(
                batch_size,
                candidate_count,
                dtype=torch.bool,
                device=query_features.device,
            )
        if candidate_similarities is None:
            candidate_similarities = F.cosine_similarity(
                query_features.unsqueeze(1), reference_features, dim=-1
            )

        query_token, query_retrieval = self.encode_query(query_features)
        element_logits = self.element_head(query_token)
        element_probabilities = F.softmax(element_logits, dim=-1)
        direct_goe_prior = element_probabilities @ self.element_goe_prior
        direct_goe_residual = 3.0 * torch.tanh(
            self.direct_goe_head(query_token).squeeze(-1)
        )
        direct_goe = (direct_goe_prior + direct_goe_residual).clamp(-5.0, 5.0)
        reference_token, reference_retrieval, reference_numeric = self.encode_reference(
            reference_features,
            reference_coarse_ids,
            reference_element_ids,
            reference_goe,
            reference_bv,
            reference_panel,
        )
        retrieval_logits = self.retrieval_scores(
            query_retrieval,
            element_logits,
            reference_retrieval,
            reference_element_ids,
        )

        pair_numeric = torch.cat(
            [
                reference_numeric,
                candidate_similarities.clamp(-1.0, 1.0).unsqueeze(-1),
            ],
            dim=-1,
        )
        expanded_query = query_token.unsqueeze(1).expand_as(reference_token)
        pair_token = self.pair_encoder(
            torch.cat(
                [
                    torch.abs(expanded_query - reference_token),
                    expanded_query * reference_token,
                    pair_numeric,
                ],
                dim=-1,
            )
        )
        citation_residual = torch.tanh(
            self.citation_head(pair_token).squeeze(-1)
        )
        citation_logits = retrieval_logits + citation_residual
        citation_weights = masked_softmax(
            citation_logits, candidate_valid_mask, dim=-1
        )

        relative_goe = 5.0 * torch.tanh(
            self.relative_goe_head(pair_token).squeeze(-1)
        )
        score_mask = candidate_valid_mask & reference_score_valid.bool()
        reference_element_confidence = element_probabilities.gather(
            1, reference_element_ids
        )
        goe_weights = (
            citation_weights
            * score_mask.to(citation_weights.dtype)
            * reference_element_confidence
        )
        goe_weight_sum = goe_weights.sum(dim=-1)
        evidence_goe = (
            goe_weights * (reference_goe + relative_goe)
        ).sum(dim=-1) / goe_weight_sum.clamp_min(1e-12)
        evidence_goe = evidence_goe.clamp(-5.0, 5.0)
        evidence_goe = torch.where(
            goe_weight_sum.gt(0), evidence_goe, torch.zeros_like(evidence_goe)
        )
        learned_goe_gate = torch.sigmoid(
            self.goe_gate_head(query_token).squeeze(-1)
        )
        goe_gate = learned_goe_gate * goe_weight_sum.clamp(0.0, 1.0)
        predicted_goe = (
            goe_gate * evidence_goe + (1.0 - goe_gate) * direct_goe
        ).clamp(-5.0, 5.0)

        return {
            "query_token": query_token,
            "query_retrieval": query_retrieval,
            "reference_retrieval": reference_retrieval,
            "retrieval_logits": retrieval_logits,
            "citation_logits": citation_logits,
            "citation_weights": citation_weights,
            "citation_residual": citation_residual,
            "relative_goe": relative_goe,
            "element_logits": element_logits,
            "element_retrieval_weight": F.softplus(
                self.element_retrieval_weight
            ),
            "direct_goe_prior": direct_goe_prior,
            "direct_goe": direct_goe,
            "evidence_goe": evidence_goe,
            "goe_gate": goe_gate,
            "reference_element_confidence": reference_element_confidence,
            "predicted_goe": predicted_goe,
            "goe_confidence": goe_weight_sum,
            "action_logit": self.action_head(query_token).squeeze(-1),
            "coarse_logits": self.coarse_head(query_token),
        }
