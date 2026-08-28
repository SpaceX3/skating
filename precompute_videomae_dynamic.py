#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from precompute_videomae_static import atomic_save, load_split_ids
from videomae_dynamic import build_dynamic_videomae_sequence


CACHE_PREFIX = "dynamic_videomae_5x8"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build five-cliplet VideoMAE dynamic caches for FS1000"
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
        "--output-dir",
        type=Path,
        default=Path("/media/v100/disk3t/skating/fs1000_dynamic_videomae_5x8"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    video_ids = load_split_ids(args.dataset_root)
    for ordinal, video_id in enumerate(video_ids, start=1):
        audio = np.load(
            args.dataset_root / "new feature/ast_feature_fs1000_new" / (video_id + ".npy"),
            mmap_mode="r",
        )
        dynamic_length = len(audio)
        output_path = args.output_dir / (
            "{}_{}_T{}.npy".format(CACHE_PREFIX, video_id, dynamic_length)
        )
        if output_path.is_file() and not args.overwrite:
            values = np.load(output_path, mmap_mode="r")
            if values.shape != (dynamic_length, 40, 768):
                raise ValueError("existing cache has wrong shape: {}".format(output_path))
            reports.append({"video_id": video_id, "dynamic_length": dynamic_length, "reused": True})
            continue

        features = np.load(args.feature_root / (video_id + ".features.npy"), mmap_mode="r")
        times = np.load(args.feature_root / (video_id + ".times.npy"), mmap_mode="r")
        sequence = build_dynamic_videomae_sequence(features, times, dynamic_length)
        atomic_save(output_path, sequence.astype(np.float16))
        reports.append({"video_id": video_id, "dynamic_length": dynamic_length, "reused": False})
        print("[{}/{}] {} T={}".format(ordinal, len(video_ids), video_id, dynamic_length), flush=True)

    summary = {
        "schema_version": "fs1000-dynamic-videomae-5x8-v1",
        "anchor_stride_seconds": 2.0,
        "cliplet_offsets_seconds": [0, 1, 2, 3, 4],
        "tokens_per_cliplet": 8,
        "tokens_per_dynamic_timestep": 40,
        "feature_dimension": 768,
        "dtype": "float16",
        "tail_policy": "repeat_last_available_cliplet",
        "videos": len(reports),
        "dynamic_timesteps": sum(item["dynamic_length"] for item in reports),
        "video_reports": reports,
    }
    (args.output_dir / "cache_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "video_reports"}, indent=2))


if __name__ == "__main__":
    main()
