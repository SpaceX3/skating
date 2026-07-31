import math

import torch
from torch import nn
import torch.nn.functional as F

from action_rag import masked_softmax


SEMANTIC_FORMAT_VERSION = "finefs-semantic-v2"
CLASSIFIER_STAGE = "finefs_semantic_classifier_v2"
GOE_STAGE = "finefs_semantic_goe_v2"


def require_semantic_v2(checkpoint, expected_stage=None):
    version = checkpoint.get("format_version")
    if version != SEMANTIC_FORMAT_VERSION:
        raise ValueError(
            "unsupported semantic checkpoint format_version {!r}; expected {!r}. "
            "v1 checkpoints cannot be used and must be retrained".format(
                version, SEMANTIC_FORMAT_VERSION
            )
        )
    stage = checkpoint.get("training_stage")
    if expected_stage is not None and stage != expected_stage:
        raise ValueError(
            "semantic checkpoint training_stage {!r}; expected {!r}".format(
                stage, expected_stage
            )
        )
    return checkpoint


def require_candidate_v2(metadata):
    version = metadata.get("format_version")
    if hasattr(version, "item"):
        version = version.item()
    if version != SEMANTIC_FORMAT_VERSION:
        raise ValueError(
            "unsupported candidate format_version {!r}; expected {!r}. "
            "v1 candidates must be regenerated".format(version, SEMANTIC_FORMAT_VERSION)
        )
    required = {"semantic_checkpoint_sha256", "corpus_version", "routing_config", "top_k"}
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError("v2 candidate metadata missing {}".format(missing))
    return metadata


def multi_positive_nll(logits, positive_mask, valid_mask=None, sample_weights=None):
    """Listwise NLL where any positive candidate is an acceptable citation."""
    positive_mask = positive_mask.bool()
    valid_mask = (
        torch.ones_like(positive_mask) if valid_mask is None else valid_mask.bool()
    )
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
    valid_mask = (
        torch.ones_like(target_weights, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.bool()
    )
    target_weights = target_weights.to(logits.dtype) * valid_mask.to(logits.dtype)
    target_sum = target_weights.sum(dim=-1)
    usable = target_sum.gt(0) & valid_mask.any(dim=-1)
    if not usable.any():
        return logits.sum() * 0.0, usable
    targets = target_weights / target_sum.unsqueeze(-1).clamp_min(1e-12)
    floor = torch.finfo(logits.dtype).min
    log_probabilities = F.log_softmax(
        logits.masked_fill(~valid_mask, floor), dim=-1
    )
    losses = -(targets * log_probabilities).sum(dim=-1)
    if sample_weights is None:
        return losses[usable].mean(), usable
    weights = sample_weights.to(losses.dtype)[usable]
    return (losses[usable] * weights).sum() / weights.sum().clamp_min(1e-12), usable


def load_classifier_state(classifier, state_dict, strict=True):
    """Load a Stage A classifier, dropping the retired retrieval projection.

    Earlier Stage A checkpoints stored a ``retrieval_encoder`` inside the
    classifier. That module never received gradients during classifier training
    because the Stage A loss only covers the action/coarse/element heads, so it
    stayed at its random initialisation and was then frozen into Stage B. The
    query-side retrieval projection now lives in :class:`ElementConditionedGOE`
    where the Stage B optimizer trains it. Those stale keys are therefore
    discarded instead of being reloaded.
    """
    retired = sorted(
        name for name in state_dict if name.startswith("retrieval_encoder.")
    )
    if retired:
        state_dict = {
            name: value
            for name, value in state_dict.items()
            if not name.startswith("retrieval_encoder.")
        }
        print(
            "dropped untrained classifier retrieval projection from checkpoint: "
            "{}".format(retired)
        )
    classifier.load_state_dict(state_dict, strict=strict)
    return retired


class OrderedInteractionEncoder(nn.Module):
    """Encode frozen frame features with first- and second-order path terms."""

    def __init__(
        self,
        input_dim,
        output_dim,
        projection_dim=16,
        time_basis=2,
        hidden_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        if projection_dim <= 0:
            raise ValueError("ordered projection_dim must be positive")
        if time_basis <= 0:
            raise ValueError("ordered time_basis must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.projection_dim = int(projection_dim)
        self.time_basis = int(time_basis)
        self.channels = self.projection_dim * self.time_basis
        self.pair_dim = self.channels * (self.channels - 1) // 2
        self.frame_projection = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.projection_dim),
            nn.GELU(),
        )
        descriptor_dim = self.channels + self.pair_dim
        self.descriptor_encoder = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), self.output_dim),
        )
        pair_indices = torch.triu_indices(
            self.channels, self.channels, offset=1
        )
        self.register_buffer("pair_rows", pair_indices[0], persistent=False)
        self.register_buffer("pair_cols", pair_indices[1], persistent=False)

    @staticmethod
    def _normalized_times(frame_times, frame_mask):
        positive_inf = torch.full_like(frame_times, torch.inf)
        negative_inf = torch.full_like(frame_times, -torch.inf)
        first = torch.where(frame_mask, frame_times, positive_inf).amin(
            dim=1, keepdim=True
        )
        last = torch.where(frame_mask, frame_times, negative_inf).amax(
            dim=1, keepdim=True
        )
        duration = (last - first).clamp_min(0.0)
        normalized = (frame_times - first) / duration.clamp_min(1e-6)
        return torch.where(frame_mask, normalized, torch.zeros_like(normalized))

    @staticmethod
    def _trapezoid_weights(normalized_times, frame_mask):
        adjacent = frame_mask[:, 1:] & frame_mask[:, :-1]
        intervals = (normalized_times[:, 1:] - normalized_times[:, :-1])
        intervals = intervals.clamp_min(0.0) * adjacent.to(normalized_times.dtype)
        weights = torch.zeros_like(normalized_times)
        weights[:, :-1] += 0.5 * intervals
        weights[:, 1:] += 0.5 * intervals
        weight_sum = weights.sum(dim=1, keepdim=True)
        uniform = frame_mask.to(normalized_times.dtype)
        uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = torch.where(weight_sum.gt(0), weights / weight_sum.clamp_min(1e-6), uniform)
        return weights * frame_mask.to(weights.dtype)

    def _legendre_basis(self, normalized_times):
        x = 2.0 * normalized_times - 1.0
        values = [torch.ones_like(x)]
        if self.time_basis > 1:
            values.append(x)
        for degree in range(2, self.time_basis):
            current = (
                (2 * degree - 1) * x * values[-1]
                - (degree - 1) * values[-2]
            ) / degree
            values.append(current)
        return torch.stack(values, dim=-1)

    def path_features(self, frame_features, frame_times, frame_mask):
        if frame_features.ndim != 3 or frame_features.shape[-1] != self.input_dim:
            raise ValueError(
                "ordered frame_features must have shape [B,T,input_dim]"
            )
        if frame_times.shape != frame_features.shape[:2]:
            raise ValueError("ordered frame_times must have shape [B,T]")
        if frame_mask.shape != frame_features.shape[:2]:
            raise ValueError("ordered frame_mask must have shape [B,T]")
        frame_mask = frame_mask.bool()
        if not frame_mask.any(dim=1).all():
            raise ValueError("every ordered query must contain at least one frame")

        frame_features = frame_features.float()
        frame_times = frame_times.to(frame_features.dtype)
        normalized_times = self._normalized_times(frame_times, frame_mask)
        weights = self._trapezoid_weights(normalized_times, frame_mask)
        projected = self.frame_projection(frame_features)
        basis = self._legendre_basis(normalized_times)
        increments = basis.unsqueeze(-1) * projected.unsqueeze(-2)
        increments = increments.flatten(start_dim=2)
        increments = increments * weights.unsqueeze(-1)

        first_order = increments.sum(dim=1)
        prefix = increments.cumsum(dim=1) - increments
        second_order = torch.einsum("bti,btj->bij", prefix, increments)
        antisymmetric = 0.5 * (
            second_order - second_order.transpose(1, 2)
        )
        ordered_pairs = antisymmetric[:, self.pair_rows, self.pair_cols]
        ordered_pairs = torch.sign(ordered_pairs) * torch.sqrt(
            ordered_pairs.abs() + 1e-8
        )
        return first_order, ordered_pairs

    def forward(self, frame_features, frame_times, frame_mask):
        first_order, ordered_pairs = self.path_features(
            frame_features, frame_times, frame_mask
        )
        return self.descriptor_encoder(
            torch.cat([first_order, ordered_pairs], dim=-1)
        )


class SemanticQueryClassifier(nn.Module):
    """Query-only action/coarse/exact-element classifier.

    This module owns classification alone. The retrieval projection used by soft
    routing lives in :class:`ElementConditionedGOE` so that it is optimized by
    the Stage B objective rather than frozen at random initialisation.
    """

    def __init__(
        self,
        coarse_classes,
        elements,
        query_dim=2048,
        evidence_dim=256,
        encoder_hidden_dim=512,
        dropout=0.1,
        ordered_enabled=False,
        ordered_projection_dim=16,
        ordered_time_basis=2,
        ordered_hidden_dim=256,
    ):
        super().__init__()
        self.query_dim = int(query_dim)
        self.evidence_dim = int(evidence_dim)
        self.coarse_classes = int(coarse_classes)
        self.elements = int(elements)
        self.ordered_enabled = bool(ordered_enabled)
        if self.ordered_enabled:
            self.query_encoder = None
            self.ordered_encoder = OrderedInteractionEncoder(
                query_dim,
                evidence_dim,
                projection_dim=ordered_projection_dim,
                time_basis=ordered_time_basis,
                hidden_dim=ordered_hidden_dim,
                dropout=dropout,
            )
        else:
            self.query_encoder = nn.Sequential(
                nn.LayerNorm(query_dim),
                nn.Linear(query_dim, encoder_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(encoder_hidden_dim, evidence_dim),
            )
            self.ordered_encoder = None
        self.action_head = nn.Linear(evidence_dim, 1)
        self.coarse_head = nn.Linear(evidence_dim, self.coarse_classes)
        self.element_head = nn.Linear(evidence_dim, self.elements)

    def classify_query(
        self,
        query_features,
        *,
        ordered_frame_features=None,
        ordered_frame_times=None,
        ordered_frame_mask=None,
    ):
        if query_features.ndim != 2 or query_features.shape[-1] != self.query_dim:
            raise ValueError("query_features must have shape [B,query_dim]")
        if self.ordered_enabled:
            if any(
                value is None
                for value in (
                    ordered_frame_features,
                    ordered_frame_times,
                    ordered_frame_mask,
                )
            ):
                raise ValueError(
                    "ordered classifier requires frame features, times, and mask"
                )
            if ordered_frame_features.shape[0] != query_features.shape[0]:
                raise ValueError("query and ordered frame batch sizes differ")
            token = self.ordered_encoder(
                ordered_frame_features,
                ordered_frame_times,
                ordered_frame_mask,
            )
        else:
            token = self.query_encoder(query_features)
        action_logit = self.action_head(token).squeeze(-1)
        coarse_logits = self.coarse_head(token)
        element_logits = self.element_head(token)
        return {
            "query_token": token,
            "action_logit": action_logit,
            "coarse_logits": coarse_logits,
            "element_logits": element_logits,
            "action_probability": torch.sigmoid(action_logit),
            "coarse_probabilities": F.softmax(coarse_logits, dim=-1),
            "element_probabilities": F.softmax(element_logits, dim=-1),
        }

    def forward(self, query_features, **kwargs):
        return self.classify_query(query_features, **kwargs)


class ElementConditionedGOE(nn.Module):
    """Visual retrieval and GOE estimation driven by frozen query predictions."""

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
        background_coarse_id=1,
    ):
        super().__init__()
        self.query_dim = int(query_dim)
        self.evidence_dim = int(evidence_dim)
        self.coarse_classes = int(coarse_classes)
        self.elements = int(elements)
        self.background_coarse_id = int(background_coarse_id)
        self.coarse_embedding = nn.Embedding(self.coarse_classes, metadata_dim)
        self.element_embedding = nn.Embedding(self.elements, metadata_dim)
        # The query-side retrieval projection is owned by this stage so that the
        # Stage B retrieval loss trains it. Keeping it in the frozen classifier
        # left it at random initialisation while still driving soft routing.
        self.query_retrieval_encoder = nn.Linear(evidence_dim, evidence_dim)
        # Retrieval is visual-only; GOE, BV, panel and labels never enter this encoder.
        self.reference_visual_encoder = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, encoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_dim, evidence_dim),
        )
        pair_dim = evidence_dim + query_dim + 2 * metadata_dim + 4
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, encoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_dim, evidence_dim),
        )
        self.citation_head = nn.Linear(evidence_dim, 1)
        self.delta_goe_head = nn.Sequential(
            nn.LayerNorm(evidence_dim), nn.Linear(evidence_dim, evidence_dim // 2),
            nn.GELU(), nn.Linear(evidence_dim // 2, 1)
        )
        self.direct_goe_head = nn.Sequential(
            nn.LayerNorm(evidence_dim), nn.Linear(evidence_dim, evidence_dim // 2),
            nn.GELU(), nn.Linear(evidence_dim // 2, 1)
        )
        self.confidence_head = nn.Sequential(nn.LayerNorm(evidence_dim), nn.Linear(evidence_dim, 1))
        self.gate_head = nn.Sequential(nn.LayerNorm(evidence_dim), nn.Linear(evidence_dim, 1))
        nn.init.constant_(self.gate_head[-1].bias, -1.0)
        self.register_buffer("element_goe_prior", torch.zeros(self.elements))
        # softplus makes every route coefficient non-negative.
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(math.expm1(1.0 / temperature))))
        self.raw_beta_element = nn.Parameter(torch.tensor(0.0))
        self.raw_beta_coarse = nn.Parameter(torch.tensor(-1.0))
        self.raw_beta_action = nn.Parameter(torch.tensor(-1.0))

    def set_element_goe_prior(self, values):
        values = torch.as_tensor(values, device=self.element_goe_prior.device, dtype=self.element_goe_prior.dtype)
        if values.shape != self.element_goe_prior.shape:
            raise ValueError("element GOE prior has the wrong shape")
        self.element_goe_prior.copy_(values)

    @staticmethod
    def _coefficient(raw, cap=100.0):
        return F.softplus(raw).clamp(max=cap)

    def encode_query_retrieval(self, query_token):
        """Project a frozen classifier token into the trainable retrieval space."""
        return F.normalize(self.query_retrieval_encoder(query_token), dim=-1)

    def encode_reference_visual(self, reference_features):
        return F.normalize(self.reference_visual_encoder(reference_features), dim=-1)

    def retrieval_scores(
        self,
        query_retrieval,
        action_probability,
        coarse_probabilities,
        element_probabilities,
        reference_retrieval,
        reference_coarse_ids,
        reference_element_ids,
        reference_is_action=None,
    ):
        if reference_retrieval.ndim == 2:
            visual = query_retrieval @ reference_retrieval.T
            element = element_probabilities[:, reference_element_ids]
            coarse = coarse_probabilities[:, reference_coarse_ids]
        elif reference_retrieval.ndim == 3:
            visual = (query_retrieval.unsqueeze(1) * reference_retrieval).sum(-1)
            element = element_probabilities.gather(1, reference_element_ids)
            coarse = coarse_probabilities.gather(1, reference_coarse_ids)
        else:
            raise ValueError("reference retrieval embeddings must be 2D or 3D")
        if reference_is_action is None:
            reference_is_action = reference_coarse_ids.ne(self.background_coarse_id)
        action_p = action_probability.unsqueeze(-1)
        action = torch.where(reference_is_action.bool(), action_p, 1.0 - action_p)
        eps = torch.finfo(visual.dtype).eps
        return (
            self._coefficient(self.raw_alpha) * visual
            + self._coefficient(self.raw_beta_element) * element.clamp_min(eps).log()
            + self._coefficient(self.raw_beta_coarse) * coarse.clamp_min(eps).log()
            + self._coefficient(self.raw_beta_action) * action.clamp_min(eps).log()
        )

    def forward(
        self,
        classifier_output,
        reference_features,
        reference_coarse_ids,
        reference_element_ids,
        reference_goe,
        reference_bv,
        reference_panel,
        reference_score_valid,
        candidate_valid_mask=None,
        candidate_similarities=None,
        reference_is_action=None,
    ):
        if reference_features.ndim != 3:
            raise ValueError("reference_features must have shape [B,K,D]")
        batch_size, candidate_count = reference_features.shape[:2]
        query_token = classifier_output["query_token"]
        if query_token.shape[0] != batch_size:
            raise ValueError("query/reference batch sizes differ")
        if candidate_valid_mask is None:
            candidate_valid_mask = torch.ones(batch_size, candidate_count, dtype=torch.bool, device=query_token.device)
        if candidate_similarities is None:
            # Raw DINO cosine is explicit pair metadata, not learned retrieval.
            raise ValueError("candidate_similarities (raw DINO cosine) is required")
        query_retrieval = self.encode_query_retrieval(query_token)
        reference_retrieval = self.encode_reference_visual(reference_features)
        retrieval_logits = self.retrieval_scores(
            query_retrieval, classifier_output["action_probability"],
            classifier_output["coarse_probabilities"], classifier_output["element_probabilities"],
            reference_retrieval, reference_coarse_ids, reference_element_ids, reference_is_action,
        )
        numeric = torch.stack([reference_goe / 5.0, reference_bv / 15.0, reference_panel / 20.0, candidate_similarities.clamp(-1, 1)], dim=-1)
        pair_token = self.pair_encoder(torch.cat([
            query_token.unsqueeze(1).expand(-1, candidate_count, -1), reference_features,
            self.coarse_embedding(reference_coarse_ids), self.element_embedding(reference_element_ids), numeric
        ], dim=-1))
        citation_logits = retrieval_logits + torch.tanh(self.citation_head(pair_token).squeeze(-1))
        citation_weights = masked_softmax(citation_logits, candidate_valid_mask, dim=-1)
        delta_goe = 5.0 * torch.tanh(self.delta_goe_head(pair_token).squeeze(-1))
        candidate_goe_unbounded = reference_goe + delta_goe
        candidate_goe = candidate_goe_unbounded.clamp(-5.0, 5.0)
        element_compatibility = classifier_output["element_probabilities"].gather(1, reference_element_ids)
        score_mask = candidate_valid_mask & reference_score_valid.bool()
        evidence_weights = citation_weights * score_mask.to(citation_weights.dtype) * element_compatibility
        evidence_weight_sum = evidence_weights.sum(-1)
        normalized = evidence_weights / evidence_weight_sum.unsqueeze(-1).clamp_min(1e-12)
        evidence_reference_goe = (normalized * reference_goe).sum(-1)
        evidence_delta_goe = (normalized * delta_goe).sum(-1)
        evidence_goe_unbounded = evidence_reference_goe + evidence_delta_goe
        evidence_goe = evidence_goe_unbounded.clamp(-5.0, 5.0)
        has_evidence = evidence_weight_sum.gt(0)
        zero = torch.zeros_like(evidence_goe)
        evidence_reference_goe = torch.where(has_evidence, evidence_reference_goe, zero)
        evidence_delta_goe = torch.where(has_evidence, evidence_delta_goe, zero)
        evidence_goe = torch.where(has_evidence, evidence_goe, zero)
        prior = classifier_output["element_probabilities"] @ self.element_goe_prior
        direct_goe = (prior + 3.0 * torch.tanh(self.direct_goe_head(query_token).squeeze(-1))).clamp(-5, 5)
        learned_confidence = torch.sigmoid(self.confidence_head(pair_token).squeeze(-1))
        goe_confidence = (normalized * learned_confidence).sum(-1) * has_evidence.to(normalized.dtype)
        goe_gate = torch.sigmoid(self.gate_head(query_token).squeeze(-1)) * goe_confidence
        predicted_goe = (goe_gate * evidence_goe + (1.0 - goe_gate) * direct_goe).clamp(-5, 5)
        return {
            **classifier_output,
            "query_retrieval": query_retrieval,
            "reference_retrieval": reference_retrieval,
            "retrieval_logits": retrieval_logits,
            "citation_logits": citation_logits,
            "citation_weights": citation_weights,
            "delta_goe": delta_goe,
            "relative_goe": delta_goe,
            "candidate_goe": candidate_goe,
            "candidate_goe_unbounded": candidate_goe_unbounded,
            "goe_evidence_weights": evidence_weights,
            "reference_element_confidence": element_compatibility,
            "direct_goe": direct_goe,
            "evidence_reference_goe": evidence_reference_goe,
            "evidence_delta_goe": evidence_delta_goe,
            "evidence_goe_unbounded": evidence_goe_unbounded,
            "evidence_goe": evidence_goe,
            "goe_confidence": goe_confidence,
            "goe_gate": goe_gate,
            "predicted_goe": predicted_goe,
        }


class SemanticPipeline(nn.Module):
    """Self-contained v2 inference pipeline with a frozen classifier boundary."""

    format_version = SEMANTIC_FORMAT_VERSION

    def __init__(self, classifier, goe_model):
        super().__init__()
        if classifier.evidence_dim != goe_model.evidence_dim:
            raise ValueError("classifier and GOE evidence dimensions differ")
        self.classifier = classifier
        self.goe_model = goe_model
        self.query_dim = classifier.query_dim
        self.evidence_dim = classifier.evidence_dim
        self.coarse_classes = classifier.coarse_classes
        self.elements = classifier.elements

    def freeze_classifier(self):
        for parameter in self.classifier.parameters():
            parameter.requires_grad = False
        self.classifier.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.classifier.parameters()):
            self.classifier.eval()
        return self

    def classify_query(self, query_features, **kwargs):
        """Classify a query and attach the trainable retrieval projection.

        ``query_retrieval`` comes from the GOE stage, so full-bank routing and the
        paired forward path share one trained projection.
        """
        output = self.classifier.classify_query(query_features, **kwargs)
        output["query_retrieval"] = self.goe_model.encode_query_retrieval(
            output["query_token"]
        )
        return output

    def encode_query_retrieval(self, query_token):
        return self.goe_model.encode_query_retrieval(query_token)

    def encode_reference_visual(self, reference_features):
        return self.goe_model.encode_reference_visual(reference_features)

    def encode_reference(self, reference_features, coarse_ids=None, element_ids=None,
                         goe=None, bv=None, panel=None):
        visual = self.encode_reference_visual(reference_features)
        numeric = None
        if goe is not None:
            numeric = torch.stack([goe / 5.0, bv / 15.0, panel / 20.0], dim=-1)
        return visual, visual, numeric

    # Small convenience surface used by full-bank mining/evaluation.
    def encode_query(self, query_features, **kwargs):
        output = self.classify_query(query_features, **kwargs)
        return output["query_token"], output["query_retrieval"]

    @property
    def element_head(self):
        return self.classifier.element_head

    def set_element_goe_prior(self, values):
        self.goe_model.set_element_goe_prior(values)

    def retrieval_scores(self, *args, **kwargs):
        return self.goe_model.retrieval_scores(*args, **kwargs)

    def forward(
        self, query_features, reference_features=None, reference_coarse_ids=None,
        reference_element_ids=None, reference_goe=None, reference_bv=None,
        reference_panel=None, reference_score_valid=None,
        ordered_frame_features=None, ordered_frame_times=None,
        ordered_frame_mask=None, **kwargs
    ):
        classification = self.classify_query(
            query_features,
            ordered_frame_features=ordered_frame_features,
            ordered_frame_times=ordered_frame_times,
            ordered_frame_mask=ordered_frame_mask,
        )
        if kwargs.get("candidate_similarities") is None and reference_features is not None:
            kwargs["candidate_similarities"] = F.cosine_similarity(
                query_features.unsqueeze(1), reference_features, dim=-1
            )
        return self.goe_model(
            classification, reference_features, reference_coarse_ids,
            reference_element_ids, reference_goe, reference_bv, reference_panel,
            reference_score_valid, **kwargs
        )


def build_semantic_pipeline(config, coarse_classes, elements, query_dim):
    classifier = SemanticQueryClassifier(
        coarse_classes, elements, query_dim,
        int(config.get("evidence_dim", 256)), int(config.get("encoder_hidden_dim", 512)),
        float(config.get("dropout", 0.1)),
        bool(config.get("ordered_enabled", False)),
        int(config.get("ordered_projection_dim", 16)),
        int(config.get("ordered_time_basis", 2)),
        int(config.get("ordered_hidden_dim", 256)),
    )
    goe = ElementConditionedGOE(
        coarse_classes, elements, query_dim,
        int(config.get("evidence_dim", 256)), int(config.get("encoder_hidden_dim", 512)),
        int(config.get("metadata_dim", 64)), float(config.get("temperature", 0.07)),
        float(config.get("dropout", 0.1)),
    )
    return SemanticPipeline(classifier, goe)


def pipeline_from_checkpoint(checkpoint, coarse_classes, elements, query_dim, device=None):
    require_semantic_v2(checkpoint, GOE_STAGE)
    model = build_semantic_pipeline(checkpoint.get("config", {}), coarse_classes, elements, query_dim)
    if "classifier_state_dict" not in checkpoint or "goe_state_dict" not in checkpoint:
        raise ValueError("v2 GOE checkpoint must contain classifier_state_dict and goe_state_dict")
    load_classifier_state(model.classifier, checkpoint["classifier_state_dict"])
    model.goe_model.load_state_dict(checkpoint["goe_state_dict"], strict=True)
    model.freeze_classifier()
    if device is not None:
        model = model.to(device)
    return model


class FineFSSemanticRAG(SemanticPipeline):
    """Deprecated construction name for the v2 split architecture.

    This does not provide v1 checkpoint compatibility; it only keeps direct model
    construction concise for callers while all saved artifacts remain v2-only.
    """

    def __init__(
        self, coarse_classes, elements, query_dim=2048, evidence_dim=256,
        encoder_hidden_dim=512, metadata_dim=64, temperature=0.07, dropout=0.1,
        ordered_enabled=False, ordered_projection_dim=16,
        ordered_time_basis=2, ordered_hidden_dim=256,
    ):
        classifier = SemanticQueryClassifier(
            coarse_classes, elements, query_dim, evidence_dim,
            encoder_hidden_dim, dropout, ordered_enabled,
            ordered_projection_dim, ordered_time_basis, ordered_hidden_dim,
        )
        goe = ElementConditionedGOE(
            coarse_classes, elements, query_dim, evidence_dim,
            encoder_hidden_dim, metadata_dim, temperature, dropout,
        )
        super().__init__(classifier, goe)
