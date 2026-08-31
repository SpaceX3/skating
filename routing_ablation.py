from __future__ import annotations

import numpy as np


def random_one_class_probabilities(
    num_queries: int,
    num_classes: int,
    seed: int,
    video_ordinal: int,
) -> np.ndarray:
    """Return deterministic one-hot class probabilities for one video.

    The video ordinal is part of the seed sequence so rebuilding one video's
    cache does not perturb routes assigned to other videos.
    """
    num_queries = int(num_queries)
    num_classes = int(num_classes)
    seed = int(seed)
    video_ordinal = int(video_ordinal)
    if num_queries <= 0:
        raise ValueError("num_queries must be positive")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if video_ordinal < 0:
        raise ValueError("video_ordinal must be non-negative")

    sequence = np.random.SeedSequence([seed, video_ordinal])
    generator = np.random.default_rng(sequence)
    indices = generator.integers(0, num_classes, size=num_queries)
    probabilities = np.zeros((num_queries, num_classes), dtype=np.float32)
    probabilities[np.arange(num_queries), indices] = 1.0
    return probabilities
