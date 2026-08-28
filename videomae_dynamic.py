from __future__ import annotations

import numpy as np


DYNAMIC_OFFSETS_SECONDS = (0.0, 1.0, 2.0, 3.0, 4.0)


def build_dynamic_videomae_sequence(
    features: np.ndarray,
    times: np.ndarray,
    dynamic_length: int,
    tolerance: float = 1e-4,
) -> np.ndarray:
    features = np.asarray(features)
    times = np.asarray(times, dtype=np.float64)
    if features.ndim != 3 or features.shape[1:] != (8, 768):
        raise ValueError("VideoMAE features must have shape [N,8,768]")
    if times.shape != (len(features), 2):
        raise ValueError("times must align with VideoMAE features")
    if int(dynamic_length) <= 0:
        raise ValueError("dynamic_length must be positive")
    starts = times[:, 0]
    if len(starts) == 0 or np.any(np.diff(starts) < 0):
        raise ValueError("VideoMAE start times must be non-empty and sorted")

    sequence = np.empty((int(dynamic_length), 40, 768), dtype=np.float32)
    for timestep in range(int(dynamic_length)):
        anchor = 2.0 * timestep
        indices = []
        for offset in DYNAMIC_OFFSETS_SECONDS:
            target = anchor + offset
            if target >= starts[-1] - tolerance:
                index = len(starts) - 1
            else:
                index = int(np.searchsorted(starts, target))
                candidates = [index]
                if index:
                    candidates.append(index - 1)
                index = min(candidates, key=lambda item: abs(starts[item] - target))
                if abs(starts[index] - target) > tolerance:
                    raise ValueError("missing VideoMAE cliplet at {:.3f}s".format(target))
            indices.append(index)
        sequence[timestep] = np.asarray(features[indices], dtype=np.float32).reshape(40, 768)
    return sequence
