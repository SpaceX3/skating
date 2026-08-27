from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


CLASS_NAMES = ("background", "jump", "spin", "sequence")


def coarse_class_index(label: str) -> int:
    if label == "background":
        return 0
    if label in {"jump", "jump_1", "jump_2", "jump_3", "jump_4"}:
        return 1
    if label == "spin":
        return 2
    if label == "sequence":
        return 3
    raise ValueError("unsupported FineFS coarse label: {}".format(label))


def _training_rows(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "train":
                yield row


def build_class_banks(
    manifest_path: Path,
    feature_root: Path,
    output_dir: Path,
) -> dict:
    counts = Counter(
        coarse_class_index(row["label"]) for row in _training_rows(manifest_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = [
        output_dir / ("{:d}_{}.tmp.npy".format(index, name))
        for index, name in enumerate(CLASS_NAMES)
    ]
    output_paths = [
        output_dir / ("{:d}_{}.npy".format(index, name))
        for index, name in enumerate(CLASS_NAMES)
    ]
    outputs = [
        np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float16, shape=(counts[index], 768)
        )
        for index, path in enumerate(temporary_paths)
    ]
    cursors = [0] * len(CLASS_NAMES)
    current_routine = None
    current_features = None
    for row in _training_rows(manifest_path):
        routine_id = str(row["routine_id"])
        if routine_id != current_routine:
            current_features = np.load(
                feature_root / (routine_id + ".features.npy"), mmap_mode="r"
            )
            current_routine = routine_id
        class_index = coarse_class_index(row["label"])
        cliplet_index = int(row["cliplet_index"])
        outputs[class_index][cursors[class_index]] = current_features[cliplet_index, 0]
        cursors[class_index] += 1

    for values in outputs:
        values.flush()
    del outputs
    for temporary_path, output_path in zip(temporary_paths, output_paths):
        os.replace(str(temporary_path), str(output_path))

    report = {
        "schema_version": "finefs-videomae-first-token-coarse-bank-v1",
        "source_split": "train",
        "class_order": list(CLASS_NAMES),
        "class_counts": {
            CLASS_NAMES[index]: int(counts[index]) for index in range(len(CLASS_NAMES))
        },
        "feature_pooling": "first_temporal_token",
        "feature_dimension": 768,
        "dtype": "float16",
        "manifest": str(manifest_path),
        "feature_root": str(feature_root),
    }
    (output_dir / "bank_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def load_class_banks(bank_dir: Path):
    return tuple(
        np.load(bank_dir / ("{:d}_{}.npy".format(index, name)), mmap_mode="r")
        for index, name in enumerate(CLASS_NAMES)
    )
