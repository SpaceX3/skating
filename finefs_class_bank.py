from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CLASS_NAMES = ("background", "jump", "spin", "sequence")
SCORE_NAMES = ("bv", "goe", "panel_score")
SCORE_DIM = len(SCORE_NAMES)
TIME_RANGE_RE = re.compile(
    r"^\s*(\d+)-(\d+(?:\.\d+)?)\s*,\s*(\d+)-(\d+(?:\.\d+)?)\s*$"
)
JUMP_RE = re.compile(r"^[1-4](?:A|T|S|Lo|F|Lz|Eu)$")


@dataclass(frozen=True)
class ScoredInterval:
    start: float
    end: float
    coarse_class: str
    descriptor: np.ndarray
    panel_corrected: bool


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


def _parse_time_range(value: str) -> tuple[float, float]:
    match = TIME_RANGE_RE.match(str(value))
    if match is None:
        raise ValueError("invalid FineFS time range: {!r}".format(value))
    start_minute, start_second, end_minute, end_second = match.groups()
    start = 60.0 * int(start_minute) + float(start_second)
    end = 60.0 * int(end_minute) + float(end_second)
    if end <= start:
        raise ValueError("non-positive FineFS time range: {!r}".format(value))
    return start, end


def _element_coarse_class(raw: dict) -> str:
    value = raw.get("coarse_class")
    coarse_class = "" if value is None else str(value).strip().lower()
    first_element = str(raw.get("element", "")).split("+")[0].strip()
    if coarse_class in {"", "none", "null"} and JUMP_RE.fullmatch(first_element):
        return "jump"
    return coarse_class


def load_scored_intervals(annotation_path: Path) -> tuple[ScoredInterval, ...]:
    with Path(annotation_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    executed = payload.get("executed_element")
    if not isinstance(executed, dict):
        raise ValueError("missing executed_element: {}".format(annotation_path))

    intervals = []
    for raw in executed.values():
        coarse_class = _element_coarse_class(raw)
        if coarse_class not in CLASS_NAMES[1:]:
            continue
        start, end = _parse_time_range(raw.get("time"))
        bv = float(raw["bv"])
        goe = float(raw["goe"])
        panel = float(raw["score_of_pannel"])
        derived_panel = bv + goe
        panel_corrected = abs(panel - derived_panel) > 0.05
        if panel_corrected:
            panel = derived_panel
        intervals.append(
            ScoredInterval(
                start=start,
                end=end,
                coarse_class=coarse_class,
                descriptor=np.asarray([bv, goe, panel], dtype=np.float32),
                panel_corrected=panel_corrected,
            )
        )
    return tuple(intervals)


def match_score_descriptor(
    intervals,
    start: float,
    end: float,
    coarse_class: str,
) -> tuple[np.ndarray, bool]:
    if coarse_class == "background":
        return np.zeros(SCORE_DIM, dtype=np.float32), False
    candidates = []
    for interval in intervals:
        if interval.coarse_class != coarse_class:
            continue
        overlap = min(float(end), interval.end) - max(float(start), interval.start)
        if overlap > 0.0:
            candidates.append((overlap, interval))
    if not candidates:
        raise ValueError(
            "no scored {} interval overlaps [{}, {}]".format(
                coarse_class, start, end
            )
        )
    _, selected = max(candidates, key=lambda item: item[0])
    return selected.descriptor.copy(), True


def build_class_banks(
    manifest_path: Path,
    feature_root: Path,
    output_dir: Path,
    annotation_dir: Path,
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
    temporary_score_paths = [
        output_dir / ("{:d}_{}_scores.tmp.npy".format(index, name))
        for index, name in enumerate(CLASS_NAMES)
    ]
    score_paths = [
        output_dir / ("{:d}_{}_scores.npy".format(index, name))
        for index, name in enumerate(CLASS_NAMES)
    ]
    temporary_mask_paths = [
        output_dir / ("{:d}_{}_score_masks.tmp.npy".format(index, name))
        for index, name in enumerate(CLASS_NAMES)
    ]
    mask_paths = [
        output_dir / ("{:d}_{}_score_masks.npy".format(index, name))
        for index, name in enumerate(CLASS_NAMES)
    ]
    outputs = [
        np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float16, shape=(counts[index], 768)
        )
        for index, path in enumerate(temporary_paths)
    ]
    score_outputs = [
        np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(counts[index], SCORE_DIM)
        )
        for index, path in enumerate(temporary_score_paths)
    ]
    mask_outputs = [
        np.lib.format.open_memmap(
            path, mode="w+", dtype=np.uint8, shape=(counts[index],)
        )
        for index, path in enumerate(temporary_mask_paths)
    ]
    cursors = [0] * len(CLASS_NAMES)
    current_routine = None
    current_features = None
    current_intervals = None
    corrected_panel_scores = 0
    for row in _training_rows(manifest_path):
        routine_id = str(row["routine_id"])
        if routine_id != current_routine:
            current_features = np.load(
                feature_root / (routine_id + ".features.npy"), mmap_mode="r"
            )
            current_intervals = load_scored_intervals(
                Path(annotation_dir) / (routine_id + ".json")
            )
            corrected_panel_scores += sum(
                int(interval.panel_corrected) for interval in current_intervals
            )
            current_routine = routine_id
        class_index = coarse_class_index(row["label"])
        cliplet_index = int(row["cliplet_index"])
        cursor = cursors[class_index]
        outputs[class_index][cursor] = current_features[cliplet_index, 0]
        descriptor, valid = match_score_descriptor(
            current_intervals,
            start=float(row["start"]),
            end=float(row["end"]),
            coarse_class=CLASS_NAMES[class_index],
        )
        score_outputs[class_index][cursor] = descriptor
        mask_outputs[class_index][cursor] = int(valid)
        cursors[class_index] += 1

    valid_score_rows = sum(int(values.sum()) for values in mask_outputs)
    if valid_score_rows <= 0:
        raise ValueError("FineFS train bank has no valid element scores")
    score_sum = np.zeros(SCORE_DIM, dtype=np.float64)
    score_square_sum = np.zeros(SCORE_DIM, dtype=np.float64)
    for scores, masks in zip(score_outputs, mask_outputs):
        valid_scores = np.asarray(scores[np.asarray(masks, dtype=bool)], dtype=np.float64)
        score_sum += valid_scores.sum(axis=0)
        score_square_sum += np.square(valid_scores).sum(axis=0)
    score_mean = score_sum / valid_score_rows
    score_variance = np.maximum(
        score_square_sum / valid_score_rows - np.square(score_mean), 1e-12
    )
    score_std = np.sqrt(score_variance)
    for scores, masks in zip(score_outputs, mask_outputs):
        valid = np.asarray(masks, dtype=bool)
        scores[valid] = (scores[valid] - score_mean) / score_std
        scores[~valid] = 0.0

    for values in outputs + score_outputs + mask_outputs:
        values.flush()
    del outputs, score_outputs, mask_outputs
    for temporary_path, output_path in zip(
        temporary_paths + temporary_score_paths + temporary_mask_paths,
        output_paths + score_paths + mask_paths,
    ):
        os.replace(str(temporary_path), str(output_path))

    report = {
        "schema_version": "finefs-videomae-first-token-scored-coarse-bank-v1",
        "source_split": "train",
        "class_order": list(CLASS_NAMES),
        "class_counts": {
            CLASS_NAMES[index]: int(counts[index]) for index in range(len(CLASS_NAMES))
        },
        "feature_pooling": "first_temporal_token",
        "feature_dimension": 768,
        "dtype": "float16",
        "score_names": list(SCORE_NAMES),
        "score_dtype": "float32",
        "score_normalization": {
            "mean": score_mean.tolist(),
            "std": score_std.tolist(),
            "population": "valid train-bank cliplets",
        },
        "score_valid_counts": {
            CLASS_NAMES[index]: int(np.load(mask_paths[index], mmap_mode="r").sum())
            for index in range(len(CLASS_NAMES))
        },
        "corrected_panel_scores": corrected_panel_scores,
        "manifest": str(manifest_path),
        "feature_root": str(feature_root),
        "annotation_dir": str(annotation_dir),
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


def load_score_banks(bank_dir: Path):
    scores = tuple(
        np.load(
            bank_dir / ("{:d}_{}_scores.npy".format(index, name)), mmap_mode="r"
        )
        for index, name in enumerate(CLASS_NAMES)
    )
    masks = tuple(
        np.load(
            bank_dir / ("{:d}_{}_score_masks.npy".format(index, name)),
            mmap_mode="r",
        )
        for index, name in enumerate(CLASS_NAMES)
    )
    return scores, masks
