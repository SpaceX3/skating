#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from class_conditioned_retrieval import (
    ClassConditionedRetriever,
    pack_cross_attention_cache,
)
from finefs_class_bank import CLASS_NAMES, SCORE_DIM, load_class_banks, load_score_banks
from precompute_videomae_static import atomic_save, load_c1_models, load_split_ids
from videomae_static import select_static_sequence_and_probabilities


TOP_CLASSES = 2
TOP_K = 4
FEATURE_DIM = 768
CACHE_DIM = (
    FEATURE_DIM
    + TOP_CLASSES * TOP_K * FEATURE_DIM
    + TOP_CLASSES * TOP_K * SCORE_DIM
    + TOP_CLASSES * TOP_K
    + TOP_CLASSES
)
CACHE_PREFIX = "static_videomae_c1_top4_score_attention"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FS1000 raw Top-4 support caches for cross-attention"
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/v100/ZYQ/FS1000 Dataset")
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("/media/v100/disk3t/finefs_pocr_classifier/features/fs1000_videomae_base_1s_stride05"),
    )
    parser.add_argument(
        "--finefs-root",
        type=Path,
        default=Path("/home/v100/ZYQ/finefs_pocr_classifier-e11-c1"),
    )
    parser.add_argument(
        "--c1-root",
        type=Path,
        default=Path("/media/v100/disk3t/finefs_pocr_classifier/experiments/e11_c1_coarse_dense05/c1"),
    )
    parser.add_argument(
        "--bank-dir",
        type=Path,
        default=Path("/media/v100/disk3t/skating/finefs_c1_scored_bank_first_token"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/media/v100/disk3t/skating/fs1000_static_videomae_c1_top4_score_attention"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--c1-batch-size", type=int, default=512)
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = load_c1_models(args.finefs_root, args.c1_root, args.device)
    score_banks, score_masks = load_score_banks(args.bank_dir)
    retriever = ClassConditionedRetriever(
        load_class_banks(args.bank_dir),
        score_banks=score_banks,
        score_masks=score_masks,
        device=args.device,
    )
    reports = []
    route_counts = Counter()
    video_ids = load_split_ids(args.dataset_root)
    for ordinal, video_id in enumerate(video_ids, start=1):
        audio = np.load(
            args.dataset_root / "new feature/ast_feature_fs1000_new" / (video_id + ".npy"),
            mmap_mode="r",
        )
        video = np.load(
            args.dataset_root / "Timesformer_output_feature_fs800" / (video_id + ".npy"),
            mmap_mode="r",
        )
        dynamic_length = min(len(audio), len(video))
        output_path = args.output_dir / (
            "{}_{}_T{}.npy".format(CACHE_PREFIX, video_id, dynamic_length)
        )
        if output_path.is_file() and not args.overwrite:
            values = np.load(output_path, mmap_mode="r")
            if values.shape != (dynamic_length, CACHE_DIM):
                raise ValueError("existing cache has the wrong shape: {}".format(output_path))
            reports.append(
                {"video_id": video_id, "dynamic_length": dynamic_length, "reused": True}
            )
            continue

        features = np.load(args.feature_root / (video_id + ".features.npy"), mmap_mode="r")
        times = np.load(args.feature_root / (video_id + ".times.npy"), mmap_mode="r")
        query, probabilities, selection_report = select_static_sequence_and_probabilities(
            features,
            times,
            dynamic_length,
            models,
            device=args.device,
            batch_size=args.c1_batch_size,
        )
        _, retrieval_details = retriever.retrieve(
            query,
            probabilities,
            top_classes=TOP_CLASSES,
            top_k=TOP_K,
            probability_power=1.0,
            query_batch_size=args.retrieval_batch_size,
        )
        cache = pack_cross_attention_cache(
            query,
            retrieval_details["selected_supports"],
            retrieval_details["selected_support_scores"],
            retrieval_details["selected_support_score_masks"],
            retrieval_details["class_weights"],
        )
        atomic_save(output_path, cache.astype(np.float16))
        for class_index in retrieval_details["selected_classes"].reshape(-1):
            route_counts[CLASS_NAMES[int(class_index)]] += 1
        reports.append({"video_id": video_id, "reused": False, **selection_report})
        print(
            "[{}/{}] {} T={}".format(ordinal, len(video_ids), video_id, dynamic_length),
            flush=True,
        )

    summary = {
        "schema_version": "fs1000-c1-top4-score-attention-cache-v1",
        "class_order": list(CLASS_NAMES),
        "routing": "C1_top2",
        "retrieval": "cosine_top4_per_routed_class",
        "static_layout": "query_768 + supports_2x4x768 + scores_2x4x3 + score_masks_2x4 + class_weights_2",
        "static_feature_dim": CACHE_DIM,
        "dtype": "float16",
        "videos": len(reports),
        "dynamic_timesteps": sum(int(report["dynamic_length"]) for report in reports),
        "selected_class_counts": dict(sorted(route_counts.items())),
        "video_reports": reports,
    }
    (args.output_dir / "cache_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "video_reports"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
