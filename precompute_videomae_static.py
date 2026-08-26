#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from videomae_static import select_static_sequence


def load_split_ids(dataset_root: Path) -> list[str]:
    result = []
    for filename in ("train_fs800.txt", "val_fs800.txt"):
        with (dataset_root / filename).open("r", encoding="utf-8") as handle:
            result.extend(line.split()[0] for line in handle if line.strip())
    if len(result) != len(set(result)):
        raise ValueError("FS1000 split files contain duplicate video ids")
    return result


def load_c1_models(finefs_root: Path, c1_root: Path, device: str):
    sys.path.insert(0, str(finefs_root / "src"))
    from finefs_pocr.c1_coarse import C1CoarseClassifier

    models = []
    for seed in (2026, 2027, 2028):
        checkpoint_path = c1_root / "runs" / ("seed" + str(seed)) / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = C1CoarseClassifier()
        model.load_state_dict(checkpoint["model"], strict=True)
        models.append(model.to(device).eval())
    return tuple(models)


def atomic_save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, values)
    os.replace(str(temporary), str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build C1-selected FS1000 VideoMAE static caches")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/v100/ZYQ/FS1000 Dataset"))
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
        "--output-dir",
        type=Path,
        default=Path("/media/v100/disk3t/skating/fs1000_static_videomae_c1"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = load_c1_models(args.finefs_root, args.c1_root, args.device)
    reports = []
    for ordinal, video_id in enumerate(load_split_ids(args.dataset_root), start=1):
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
            "static_videomae_c1_{}_T{}.npy".format(video_id, dynamic_length)
        )
        if output_path.is_file() and not args.overwrite:
            values = np.load(output_path, mmap_mode="r")
            if values.shape != (dynamic_length, 768):
                raise ValueError("existing cache has the wrong shape: {}".format(output_path))
            reports.append({"video_id": video_id, "dynamic_length": dynamic_length, "reused": True})
            continue
        feature_path = args.feature_root / (video_id + ".features.npy")
        time_path = args.feature_root / (video_id + ".times.npy")
        features = np.load(feature_path, mmap_mode="r")
        times = np.load(time_path, mmap_mode="r")
        sequence, report = select_static_sequence(
            features,
            times,
            dynamic_length,
            models,
            device=args.device,
            batch_size=args.batch_size,
        )
        atomic_save(output_path, sequence)
        reports.append({"video_id": video_id, "reused": False, **report})
        print("[{}/811] {} T={}".format(ordinal, video_id, dynamic_length), flush=True)

    offset_counts = Counter()
    for report in reports:
        offset_counts.update(report.get("selected_offset_counts", {}))
    summary = {
        "schema_version": "fs1000-static-videomae-c1-v1",
        "videos": len(reports),
        "seeds": [2026, 2027, 2028],
        "candidate_offsets_seconds": [0.0, 0.5, 1.0, 1.5],
        "first_cliplet_pooling": "mean_8_temporal_tokens",
        "selected_offset_counts": dict(sorted(offset_counts.items())),
        "incomplete_candidate_groups": sum(
            int(report.get("incomplete_candidate_groups", 0)) for report in reports
        ),
        "previous_vector_fallbacks": sum(
            int(report.get("previous_vector_fallbacks", 0)) for report in reports
        ),
        "dynamic_timesteps": sum(int(report["dynamic_length"]) for report in reports),
        "video_reports": reports,
    }
    report_path = args.output_dir / "cache_report.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "video_reports"}, indent=2))


if __name__ == "__main__":
    main()
