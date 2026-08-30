from __future__ import annotations

import numpy as np
import torch


def pack_cross_attention_cache(
    query: np.ndarray,
    selected_supports: np.ndarray,
    selected_support_scores: np.ndarray,
    selected_support_score_masks: np.ndarray,
    class_weights: np.ndarray,
) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    selected_supports = np.asarray(selected_supports, dtype=np.float32)
    selected_support_scores = np.asarray(selected_support_scores, dtype=np.float32)
    selected_support_score_masks = np.asarray(
        selected_support_score_masks, dtype=np.float32
    )
    class_weights = np.asarray(class_weights, dtype=np.float32)
    if query.ndim != 2 or selected_supports.ndim != 4:
        raise ValueError("query and selected_supports must be [B,D] and [B,C,K,D]")
    batch, classes, _, dim = selected_supports.shape
    if query.shape != (batch, dim) or class_weights.shape != (batch, classes):
        raise ValueError("query, supports, and class weights do not align")
    if selected_support_scores.shape[:3] != selected_supports.shape[:3]:
        raise ValueError("support scores must align with selected supports")
    if selected_support_score_masks.shape != selected_supports.shape[:3]:
        raise ValueError("support score masks must align with selected supports")
    weights = np.maximum(class_weights, 0.0)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return np.concatenate(
        (
            query,
            selected_supports.reshape(batch, -1),
            selected_support_scores.reshape(batch, -1),
            selected_support_score_masks.reshape(batch, -1),
            weights,
        ),
        axis=1,
    )


def _normalize_tensor_rows(values: torch.Tensor) -> torch.Tensor:
    return values / values.norm(dim=1, keepdim=True).clamp_min(1e-12)


class ClassConditionedRetriever:
    """Cosine Top-K retriever with one immutable bank per coarse class."""

    def __init__(
        self,
        class_banks,
        score_banks=None,
        score_masks=None,
        device: str = "cuda:0",
    ):
        if not class_banks:
            raise ValueError("at least one class bank is required")
        self.device = torch.device(device)
        self.class_banks = tuple(np.asarray(bank) for bank in class_banks)
        if (score_banks is None) != (score_masks is None):
            raise ValueError("score_banks and score_masks must be provided together")
        self.score_banks = (
            None if score_banks is None else tuple(np.asarray(bank) for bank in score_banks)
        )
        self.score_masks = (
            None if score_masks is None else tuple(np.asarray(mask) for mask in score_masks)
        )
        feature_dim = self.class_banks[0].shape[1]
        self.normalized_banks = []
        for bank in self.class_banks:
            if bank.ndim != 2 or bank.shape[1] != feature_dim or len(bank) == 0:
                raise ValueError("all class banks must be non-empty [N,D] matrices")
            values = torch.from_numpy(np.asarray(bank, dtype=np.float32)).to(self.device)
            self.normalized_banks.append(_normalize_tensor_rows(values))
        if self.score_banks is not None:
            if len(self.score_banks) != len(self.class_banks):
                raise ValueError("score banks must match the number of feature banks")
            score_dim = self.score_banks[0].shape[1]
            for features, scores, masks in zip(
                self.class_banks, self.score_banks, self.score_masks
            ):
                if scores.shape != (len(features), score_dim):
                    raise ValueError("each score bank must align with its feature bank")
                if masks.shape != (len(features),):
                    raise ValueError("each score mask must align with its feature bank")

    def retrieve(
        self,
        query: np.ndarray,
        class_probabilities: np.ndarray,
        top_classes: int = 2,
        top_k: int = 4,
        temperature: float = 0.1,
        probability_power: float = 1.0,
        query_batch_size: int = 256,
    ) -> tuple[np.ndarray, dict]:
        query = np.asarray(query, dtype=np.float32)
        class_probabilities = np.asarray(class_probabilities, dtype=np.float32)
        classes = len(self.class_banks)
        if query.ndim != 2 or query.shape[1] != self.class_banks[0].shape[1]:
            raise ValueError("query must have shape [B,D] matching the banks")
        if class_probabilities.shape != (len(query), classes):
            raise ValueError("class_probabilities must have shape [B,C]")
        if not 1 <= int(top_classes) <= classes:
            raise ValueError("top_classes is outside the available class range")
        if int(top_k) <= 0 or int(query_batch_size) <= 0:
            raise ValueError("top_k and query_batch_size must be positive")
        if any(len(bank) < int(top_k) for bank in self.class_banks):
            raise ValueError("top_k exceeds a class bank size")

        selected = np.argsort(-class_probabilities, axis=1)[:, : int(top_classes)]
        batch_size, dim = query.shape
        retrieved = np.zeros(
            (batch_size, classes, int(top_k), dim), dtype=np.float32
        )
        similarities = np.zeros(
            (batch_size, classes, int(top_k)), dtype=np.float32
        )
        retrieved_scores = None
        retrieved_score_masks = None
        if self.score_banks is not None:
            score_dim = self.score_banks[0].shape[1]
            retrieved_scores = np.zeros(
                (batch_size, classes, int(top_k), score_dim), dtype=np.float32
            )
            retrieved_score_masks = np.zeros(
                (batch_size, classes, int(top_k)), dtype=np.float32
            )
        normalized_query = _normalize_tensor_rows(
            torch.from_numpy(query).to(self.device)
        )
        with torch.inference_mode():
            for class_index, bank_tensor in enumerate(self.normalized_banks):
                rows = np.flatnonzero(np.any(selected == class_index, axis=1))
                for begin in range(0, len(rows), int(query_batch_size)):
                    batch_rows = rows[begin : begin + int(query_batch_size)]
                    scores = normalized_query[batch_rows] @ bank_tensor.T
                    values, indices = torch.topk(scores, k=int(top_k), dim=1)
                    selected_indices = indices.cpu().numpy()
                    retrieved[batch_rows, class_index] = np.asarray(
                        self.class_banks[class_index][selected_indices],
                        dtype=np.float32,
                    )
                    if retrieved_scores is not None:
                        retrieved_scores[batch_rows, class_index] = np.asarray(
                            self.score_banks[class_index][selected_indices],
                            dtype=np.float32,
                        )
                        retrieved_score_masks[batch_rows, class_index] = np.asarray(
                            self.score_masks[class_index][selected_indices],
                            dtype=np.float32,
                        )
                    similarities[batch_rows, class_index] = values.float().cpu().numpy()

        fused, details = aggregate_retrieved_vectors(
            query,
            retrieved,
            similarities,
            class_probabilities,
            top_classes=top_classes,
            temperature=temperature,
            probability_power=probability_power,
        )
        details["top_k"] = int(top_k)
        details["top_classes"] = int(top_classes)
        details["selected_supports"] = np.take_along_axis(
            retrieved, details["selected_classes"][..., None, None], axis=1
        )
        if retrieved_scores is not None:
            details["selected_support_scores"] = np.take_along_axis(
                retrieved_scores,
                details["selected_classes"][..., None, None],
                axis=1,
            )
            details["selected_support_score_masks"] = np.take_along_axis(
                retrieved_score_masks,
                details["selected_classes"][..., None],
                axis=1,
            )
        return fused, details


def aggregate_retrieved_vectors(
    query: np.ndarray,
    retrieved: np.ndarray,
    similarities: np.ndarray,
    class_probabilities: np.ndarray,
    top_classes: int = 2,
    temperature: float = 0.1,
    probability_power: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Fuse query with class-conditioned Top-K vectors.

    Shapes are query [B,D], retrieved [B,C,K,D], similarities [B,C,K], and
    class_probabilities [B,C]. The output is concat(query, fused_knowledge).
    """
    query = np.asarray(query, dtype=np.float32)
    retrieved = np.asarray(retrieved, dtype=np.float32)
    similarities = np.asarray(similarities, dtype=np.float32)
    class_probabilities = np.asarray(class_probabilities, dtype=np.float32)
    if query.ndim != 2 or retrieved.ndim != 4 or similarities.ndim != 3:
        raise ValueError("invalid query/retrieval dimensions")
    batch, classes, k, dim = retrieved.shape
    if query.shape != (batch, dim):
        raise ValueError("query and retrieved feature dimensions do not match")
    if similarities.shape != (batch, classes, k):
        raise ValueError("similarities must match retrieved [B,C,K]")
    if class_probabilities.shape != (batch, classes):
        raise ValueError("class_probabilities must have shape [B,C]")
    if not 1 <= int(top_classes) <= classes or float(temperature) <= 0:
        raise ValueError("top_classes or temperature is invalid")
    if float(probability_power) <= 0:
        raise ValueError("probability_power must be positive")

    selected = np.argsort(-class_probabilities, axis=1)[:, : int(top_classes)]
    class_representations = np.empty((batch, classes, dim), dtype=np.float32)
    for class_index in range(classes):
        logits = similarities[:, class_index] / float(temperature)
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        class_representations[:, class_index] = np.sum(
            retrieved[:, class_index] * weights[..., None], axis=1
        )

    selected_probabilities = np.take_along_axis(
        class_probabilities, selected, axis=1
    )
    selected_probabilities = np.power(np.maximum(selected_probabilities, 0.0), probability_power)
    selected_probabilities /= np.maximum(
        selected_probabilities.sum(axis=1, keepdims=True), 1e-12
    )
    selected_representations = np.take_along_axis(
        class_representations, selected[..., None], axis=1
    )
    fused_knowledge = np.sum(
        selected_representations * selected_probabilities[..., None], axis=1
    )
    return np.concatenate((query, fused_knowledge), axis=1), {
        "selected_classes": selected,
        "class_weights": selected_probabilities,
    }
