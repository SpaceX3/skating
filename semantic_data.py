import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.utils.data as data


def make_video_split(corpus, seed=2026, train_ratio=0.70, val_ratio=0.15):
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("require 0 < train_ratio, val_ratio and train_ratio + val_ratio < 1")
    video_ids = sorted(set(str(value) for value in corpus["video_ids"]))
    if len(video_ids) < 3:
        raise ValueError("semantic splitting requires at least three videos")
    generator = random.Random(int(seed))
    generator.shuffle(video_ids)
    train_count = max(1, int(round(len(video_ids) * train_ratio)))
    val_count = max(1, int(round(len(video_ids) * val_ratio)))
    if train_count + val_count >= len(video_ids):
        val_count = 1
        train_count = len(video_ids) - 2
    splits = {
        "train": sorted(video_ids[:train_count]),
        "val": sorted(video_ids[train_count : train_count + val_count]),
        "test": sorted(video_ids[train_count + val_count :]),
    }
    manifest = {
        "format_version": "finefs-semantic-split-v1",
        "corpus_version": str(corpus["metadata_version"]),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "source_videos": len(video_ids),
        "splits": splits,
    }
    validate_video_split(corpus, manifest)
    return manifest


def save_video_split(path, manifest, overwrite=False):
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            "split already exists; pass --overwrite-split to replace it: {}".format(path)
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def load_video_split(path, corpus):
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_video_split(corpus, manifest)
    return manifest


def validate_video_split(corpus, manifest):
    if manifest.get("format_version") != "finefs-semantic-split-v1":
        raise ValueError("unsupported FineFS semantic split format")
    if str(manifest.get("corpus_version")) != str(corpus["metadata_version"]):
        raise ValueError("split and action corpus metadata versions differ")
    split_sets = {
        name: set(str(value) for value in manifest["splits"][name])
        for name in ("train", "val", "test")
    }
    if any(not values for values in split_sets.values()):
        raise ValueError("train/val/test must all contain videos")
    if split_sets["train"] & split_sets["val"]:
        raise ValueError("train and val videos overlap")
    if split_sets["train"] & split_sets["test"]:
        raise ValueError("train and test videos overlap")
    if split_sets["val"] & split_sets["test"]:
        raise ValueError("val and test videos overlap")
    corpus_videos = set(str(value) for value in corpus["video_ids"])
    assigned = set().union(*split_sets.values())
    if assigned != corpus_videos:
        raise ValueError(
            "split videos differ from corpus videos: missing={} extra={}".format(
                len(corpus_videos - assigned), len(assigned - corpus_videos)
            )
        )


def split_statistics(corpus, manifest):
    video_ids = [str(value) for value in corpus["video_ids"]]
    coarse_vocab = list(corpus["coarse_class_vocab"])
    result = {}
    for split_name in ("train", "val", "test"):
        selected = set(manifest["splits"][split_name])
        indices = [i for i, video_id in enumerate(video_ids) if video_id in selected]
        coarse = Counter(
            coarse_vocab[int(corpus["coarse_class_ids"][index])]
            for index in indices
        )
        result[split_name] = {
            "videos": len(selected),
            "prototypes": len(indices),
            "scored_action_prototypes": int(
                corpus["valid_score_mask"][indices].sum().item()
            ),
            "coarse_counts": dict(sorted(coarse.items())),
        }
    return result


class FineFSSemanticDataset(data.Dataset):
    def __init__(self, corpus, video_ids):
        selected = set(str(value) for value in video_ids)
        background_id = list(corpus["coarse_class_vocab"]).index("background")
        self.indices = []
        for index, video_id in enumerate(corpus["video_ids"]):
            if str(video_id) not in selected:
                continue
            coarse_id = int(corpus["coarse_class_ids"][index])
            if bool(corpus["valid_score_mask"][index]) or coarse_id == background_id:
                self.indices.append(index)
        if not self.indices:
            raise ValueError("FineFS semantic dataset is empty")

    def __getitem__(self, index):
        return self.indices[index]

    def __len__(self):
        return len(self.indices)


class SemanticCandidateSampler:
    """Balanced O(K) candidate sampling with same-video exclusion."""

    def __init__(
        self,
        corpus,
        train_video_ids,
        candidate_count=32,
        positive_count=4,
        candidate_table_count=2,
        seed=2026,
    ):
        if candidate_count < 4:
            raise ValueError("candidate_count must be at least 4")
        if positive_count <= 0 or positive_count >= candidate_count:
            raise ValueError("require 0 < positive_count < candidate_count")
        self.corpus = corpus
        self.candidate_count = int(candidate_count)
        self.positive_count = int(positive_count)
        self.candidate_table_count = int(candidate_table_count)
        if self.candidate_table_count <= 0:
            raise ValueError("candidate_table_count must be positive")
        self.generator = np.random.default_rng(int(seed))
        video_vocab = {
            value: index
            for index, value in enumerate(sorted(set(str(v) for v in corpus["video_ids"])))
        }
        self.video_ids = np.asarray(
            [video_vocab[str(value)] for value in corpus["video_ids"]],
            dtype=np.int64,
        )
        self.element_ids = corpus["element_ids"].long().numpy()
        self.coarse_ids = corpus["coarse_class_ids"].long().numpy()
        train_videos = set(str(value) for value in train_video_ids)
        self.reference_indices = np.asarray(
            [
                index
                for index, video_id in enumerate(corpus["video_ids"])
                if str(video_id) in train_videos
            ],
            dtype=np.int64,
        )
        if self.reference_indices.size == 0:
            raise ValueError("the semantic train reference bank is empty")
        self.by_element = defaultdict(list)
        self.by_coarse = defaultdict(list)
        for index in self.reference_indices:
            self.by_element[self.element_ids[index]].append(index)
            self.by_coarse[self.coarse_ids[index]].append(index)
        self.by_element = {
            key: np.asarray(values, dtype=np.int64)
            for key, values in self.by_element.items()
        }
        self.by_coarse = {
            key: np.asarray(values, dtype=np.int64)
            for key, values in self.by_coarse.items()
        }
        self._table_cursor = 0
        self.candidate_tables = self._build_candidate_tables()

    def _build_candidate_tables(self):
        tables = np.full(
            (
                self.candidate_table_count,
                len(self.video_ids),
                self.candidate_count,
            ),
            -1,
            dtype=np.int64,
        )
        query_indices = self.reference_indices.tolist()
        for table_index in range(self.candidate_table_count):
            for query_index in query_indices:
                tables[table_index, query_index] = self.sample_one(query_index)
        return torch.from_numpy(tables)

    def to(self, device):
        self.candidate_tables = self.candidate_tables.to(device)
        return self

    def _draw(self, pool, count, query_video, used, excluded_element=None):
        """Draw a small unique subset without scanning the full reference bank."""
        if count <= 0 or pool is None or len(pool) == 0:
            return []
        selected = []
        max_draws = max(64, count * 16)
        draws = 0
        while len(selected) < count and draws < max_draws:
            draw_count = min(max((count - len(selected)) * 4, 8), max_draws - draws)
            candidates = self.generator.choice(pool, size=draw_count, replace=True)
            draws += draw_count
            for value in candidates:
                index = int(value)
                if self.video_ids[index] == query_video or index in used:
                    continue
                if (
                    excluded_element is not None
                    and self.element_ids[index] == excluded_element
                ):
                    continue
                used.add(index)
                selected.append(index)
                if len(selected) == count:
                    break
        return selected

    def sample_one(self, query_index):
        query_video = self.video_ids[query_index]
        query_element = self.element_ids[query_index]
        query_coarse = self.coarse_ids[query_index]
        used = set()
        selected = self._draw(
            self.by_element.get(query_element),
            self.positive_count,
            query_video,
            used,
        )

        same_coarse_target = max(self.candidate_count // 2 - len(selected), 0)
        additions = self._draw(
            self.by_coarse.get(query_coarse),
            same_coarse_target,
            query_video,
            used,
            excluded_element=query_element,
        )
        selected.extend(additions)

        remaining = self.candidate_count - len(selected)
        additions = self._draw(
            self.reference_indices, remaining, query_video, used
        )
        selected.extend(additions)

        if len(selected) < self.candidate_count:
            # This path is only relevant to very small synthetic corpora.
            fallback = self.reference_indices[
                self.video_ids[self.reference_indices] != query_video
            ]
            if fallback.size == 0:
                raise ValueError(
                    "no cross-video semantic candidates for query {}".format(query_index)
                )
            while len(selected) < self.candidate_count:
                selected.append(int(self.generator.choice(fallback)))
        self.generator.shuffle(selected)
        return selected

    def sample(self, query_indices):
        query_indices = torch.as_tensor(
            query_indices,
            dtype=torch.long,
            device=self.candidate_tables.device,
        )
        table_index = self._table_cursor % self.candidate_table_count
        self._table_cursor += 1
        return self.candidate_tables[table_index, query_indices]


def prototype_sample_weights(corpus, query_indices):
    if "prototype_counts" in corpus:
        counts = corpus["prototype_counts"][query_indices].float()
    else:
        instance_counts = Counter(corpus["instance_ids"])
        counts = torch.tensor(
            [instance_counts[corpus["instance_ids"][int(i)]] for i in query_indices],
            dtype=torch.float32,
        )
    return counts.clamp_min(1.0).reciprocal()


def closed_set_element_ids(corpus, train_video_ids):
    train_videos = set(str(value) for value in train_video_ids)
    return set(
        int(corpus["element_ids"][index])
        for index, video_id in enumerate(corpus["video_ids"])
        if str(video_id) in train_videos
        and bool(corpus["valid_score_mask"][index])
    )
