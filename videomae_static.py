from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


CANDIDATE_OFFSETS = (0.0, 0.5, 1.0, 1.5)


def build_timestep_candidates(
    times: np.ndarray,
    timestep_index: int,
    tolerance: float = 1e-6,
) -> list[dict]:
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 2 or times.shape[1] != 2 or not np.isfinite(times).all():
        raise ValueError("times must contain finite [start,end] pairs")
    if int(timestep_index) < 0:
        raise ValueError("timestep_index must be non-negative")
    anchor = 2.0 * int(timestep_index)
    candidates = []
    for offset in CANDIDATE_OFFSETS:
        start = anchor + offset
        selected = np.flatnonzero(
            (times[:, 0] >= start - tolerance)
            & (times[:, 1] <= start + 5.0 + tolerance)
        ).astype(np.int64)
        target = selected[times[selected, 1] <= start + 2.0 + tolerance]
        context = selected[times[selected, 1] > start + 2.0 + tolerance]
        if len(target) != 3 or len(context) != 6:
            continue
        candidates.append(
            {
                "start": float(start),
                "offset": float(offset),
                "first_index": int(target[0]),
                "target_indices": target,
                "context_indices": context,
            }
        )
    return candidates


def choose_by_ensemble_confidence(
    seed_logits: np.ndarray,
) -> tuple[int, np.ndarray]:
    seed_logits = np.asarray(seed_logits, dtype=np.float64)
    if (
        seed_logits.ndim != 3
        or min(seed_logits.shape) <= 0
        or not np.isfinite(seed_logits).all()
    ):
        raise ValueError("seed_logits must have shape [seeds,candidates,classes]")
    logits = seed_logits.mean(axis=0)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    confidence = probabilities.max(axis=1)
    return int(np.argmax(confidence)), probabilities


def _score_candidates(
    features: np.ndarray,
    candidates: Sequence[dict],
    models: Sequence[torch.nn.Module],
    device: str,
    batch_size: int,
) -> np.ndarray:
    if not candidates:
        return np.empty((len(models), 0, 0), dtype=np.float32)
    if not models:
        raise ValueError("at least one C1 model is required")
    torch_device = torch.device(device)
    models = tuple(model.to(torch_device).eval() for model in models)
    outputs = [[] for _ in models]
    with torch.inference_mode():
        for begin in range(0, len(candidates), int(batch_size)):
            batch = candidates[begin : begin + int(batch_size)]
            target = np.stack(
                [features[item["target_indices"]].reshape(-1, features.shape[-1]) for item in batch]
            ).astype(np.float32, copy=False)
            context = np.stack(
                [features[item["context_indices"]].reshape(-1, features.shape[-1]) for item in batch]
            ).astype(np.float32, copy=False)
            target_tensor = torch.from_numpy(target).to(torch_device)
            context_tensor = torch.from_numpy(context).to(torch_device)
            for model_index, model in enumerate(models):
                logits = model(target_tensor, context_tensor)
                outputs[model_index].append(logits.float().cpu().numpy())
    return np.stack(
        [np.concatenate(seed_output, axis=0) for seed_output in outputs], axis=0
    )


def select_static_sequence(
    features: np.ndarray,
    times: np.ndarray,
    dynamic_length: int,
    models: Sequence[torch.nn.Module],
    device: str = "cuda:0",
    batch_size: int = 512,
) -> tuple[np.ndarray, dict]:
    features = np.asarray(features)
    times = np.asarray(times)
    if features.ndim != 3 or len(features) != len(times):
        raise ValueError("features and times must align as [cliplets,tokens,hidden]")
    if int(dynamic_length) <= 0 or int(batch_size) <= 0:
        raise ValueError("dynamic_length and batch_size must be positive")

    groups = [
        build_timestep_candidates(times, timestep_index)
        for timestep_index in range(int(dynamic_length))
    ]
    flat_candidates = [candidate for group in groups for candidate in group]
    seed_logits = _score_candidates(
        features, flat_candidates, models, device=device, batch_size=batch_size
    )
    sequence = np.empty((int(dynamic_length), features.shape[-1]), dtype=np.float32)
    selected_offset_counts = {"0.0": 0, "0.5": 0, "1.0": 0, "1.5": 0}
    previous_vector_fallbacks = 0
    incomplete_candidate_groups = 0
    cursor = 0
    for timestep_index, candidates in enumerate(groups):
        count = len(candidates)
        incomplete_candidate_groups += int(count < len(CANDIDATE_OFFSETS))
        if count:
            selected, _ = choose_by_ensemble_confidence(
                seed_logits[:, cursor : cursor + count]
            )
            candidate = candidates[selected]
            sequence[timestep_index] = np.asarray(
                features[candidate["first_index"], 0], dtype=np.float32
            )
            selected_offset_counts["{:.1f}".format(candidate["offset"])] += 1
            cursor += count
        elif timestep_index:
            sequence[timestep_index] = sequence[timestep_index - 1]
            previous_vector_fallbacks += 1
        else:
            raise ValueError("the first dynamic timestep has no complete C1 candidate")
    return sequence, {
        "dynamic_length": int(dynamic_length),
        "candidate_windows": len(flat_candidates),
        "incomplete_candidate_groups": incomplete_candidate_groups,
        "previous_vector_fallbacks": previous_vector_fallbacks,
        "selected_offset_counts": selected_offset_counts,
    }
