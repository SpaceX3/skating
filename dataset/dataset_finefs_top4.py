"""FineFS dataset adapter for the Top-4 Cross-Attention AQA experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from dataset.dataset_fs800 import av_collate_fn_with_static


def _numeric_sort_key(value: str):
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


class FineFSFeatureDatasetWithStaticCache(data.Dataset):
    """Load one complete FineFS routine and its per-timestep static cache."""

    def __init__(
        self,
        split_json,
        annotation_dir,
        audio_feature_dir,
        video_feature_dir,
        static_cache_dir,
        static_cache_prefix="static_videomae_c1_top4_cross_attention",
        static_feature_dim=6914,
        split="train",
    ):
        split_data = json.loads(Path(split_json).read_text(encoding="utf-8"))
        video_to_split = split_data.get("video_to_split")
        if not isinstance(video_to_split, dict):
            raise ValueError("split JSON must contain a video_to_split object")

        self.annotation_dir = Path(annotation_dir)
        self.audio_feature_dir = Path(audio_feature_dir)
        self.video_feature_dir = Path(video_feature_dir)
        self.static_cache_dir = Path(static_cache_dir)
        self.static_cache_prefix = str(static_cache_prefix)
        self.static_feature_dim = int(static_feature_dim)
        self.split = str(split)
        self.video_ids = sorted(
            [video_id for video_id, video_split in video_to_split.items() if video_split == split],
            key=_numeric_sort_key,
        )

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        audio_feature = torch.from_numpy(
            np.load(self.audio_feature_dir / (video_id + ".npy"))
        )
        video_feature = torch.from_numpy(
            np.load(self.video_feature_dir / (video_id + ".npy"))
        )
        if audio_feature.ndim != 2 or audio_feature.shape[-1] != 768:
            raise ValueError(
                "FineFS AST feature {} must have shape [T,768], got {}".format(
                    video_id, tuple(audio_feature.shape)
                )
            )
        if video_feature.ndim != 3 or video_feature.shape[-1] != 768:
            raise ValueError(
                "FineFS Timesformer feature {} must have shape [T,N,768], got {}".format(
                    video_id, tuple(video_feature.shape)
                )
            )

        dynamic_length = min(audio_feature.shape[0], video_feature.shape[0])
        static_path = self.static_cache_dir / (
            "{}_{}_T{}.npy".format(
                self.static_cache_prefix, video_id, dynamic_length
            )
        )
        if not static_path.is_file():
            raise FileNotFoundError("missing static feature cache: {}".format(static_path))
        static_values = np.load(static_path)
        expected_shape = (dynamic_length, self.static_feature_dim)
        if static_values.shape != expected_shape:
            raise ValueError(
                "static feature {} has shape {}, expected {}".format(
                    static_path, static_values.shape, expected_shape
                )
            )
        static_feature = torch.from_numpy(static_values).float()

        annotation = json.loads(
            (self.annotation_dir / (video_id + ".json")).read_text(encoding="utf-8")
        )
        factor = float(annotation["factor"])
        tes = float(annotation["total_element_score"])
        pcs = float(annotation["total_program_component_score(factored)"]) / factor
        components = annotation["program_component"]

        return (
            audio_feature,
            video_feature,
            tes,
            pcs,
            float(components["skating_skills"]["score_of_pannel"]),
            float(components["transitions"]["score_of_pannel"]),
            float(components["performance"]["score_of_pannel"]),
            float(components["composition"]["score_of_pannel"]),
            float(components["interpretation"]["score_of_pannel"]),
            static_feature,
            video_id,
        )


__all__ = ["FineFSFeatureDatasetWithStaticCache", "av_collate_fn_with_static"]
