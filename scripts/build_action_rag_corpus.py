import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F


COARSE_CLASSES = ["unknown", "background", "jump", "spin", "sequence"]
PROTOTYPE_LIMITS = {"jump": 3, "spin": 4, "sequence": 8, "unknown": 4}


def rounded(values):
    return tuple(round(float(value), 4) for value in values)


def annotation_signature(annotation):
    components = annotation["program_component"]
    return rounded(
        [
            annotation["total_element_score"],
            annotation["total_program_component_score(factored)"],
            components["skating_skills"]["score_of_pannel"],
            components["transitions"]["score_of_pannel"],
            components["performance"]["score_of_pannel"],
            components["composition"]["score_of_pannel"],
            components["interpretation"]["score_of_pannel"],
            annotation["factor"],
        ]
    )


def validation_signatures(fs1000_root):
    path = os.path.join(fs1000_root, "val_fs800.txt")
    with open(path, "r", encoding="utf-8") as handle:
        return {rounded(line.split()[1:9]) for line in handle if line.strip()}


def parse_time(value):
    if not isinstance(value, str) or "," not in value:
        raise ValueError("invalid action time {!r}".format(value))

    def seconds(part):
        minute, second = part.strip().split("-", 1)
        return 60.0 * float(minute) + float(second)

    start_text, end_text = value.split(",", 1)
    start, end = seconds(start_text), seconds(end_text)
    if end <= start:
        raise ValueError("non-positive action interval {!r}".format(value))
    return start, end


def canonical_element(value):
    return re.sub(r"\s+", "", str(value or "")).upper()


def infer_coarse(element, coarse):
    coarse = str(coarse or "").strip().lower()
    if coarse in ("jump", "spin", "sequence"):
        return coarse
    normalized = canonical_element(element)
    if "SQ" in normalized:
        return "sequence"
    if "SP" in normalized:
        return "spin"
    if re.search(r"(^|\+)[1-4](A|T|S|LO|F|LZ)", normalized):
        return "jump"
    return "unknown"


def trimmed_judge_grade(element):
    scores = element.get("judge_score")
    if not isinstance(scores, list) or len(scores) < 3:
        return float("nan")
    values = sorted(float(score) for score in scores)
    return float(np.mean(values[1:-1]))


def find_cache_pair(cache_dir, cache_prefix, video_id):
    pattern = os.path.join(cache_dir, f"{cache_prefix}_{video_id}_T*.npy")
    feature_paths = [path for path in glob.glob(pattern) if not path.endswith(".times.npy")]
    if len(feature_paths) != 1:
        raise ValueError(
            f"expected one feature cache for FineFS video {video_id}, got {feature_paths}"
        )
    feature_path = feature_paths[0]
    times_path = feature_path[:-4] + ".times.npy"
    if not os.path.exists(times_path):
        raise FileNotFoundError(times_path)
    return feature_path, times_path


def select_prototypes(features, times, start, end, limit):
    sample_start = times[:, 2]
    sample_end = times[:, 3]
    overlap = np.maximum(0.0, np.minimum(sample_end, end) - np.maximum(sample_start, start))
    indices = np.flatnonzero(overlap > 0)
    if len(indices) == 0:
        centers = (sample_start + sample_end) / 2.0
        action_center = 0.5 * (start + end)
        indices = np.asarray([int(np.argmin(np.abs(centers - action_center)))])
    bins = np.array_split(indices, min(limit, len(indices)))
    return [features[group].mean(axis=0) for group in bins if len(group)]


def build_corpus(args):
    val_signatures = validation_signatures(args.fs1000_root)
    annotation_paths = sorted(
        glob.glob(os.path.join(args.finefs_root, "annotation", "*.json")),
        key=lambda path: int(os.path.splitext(os.path.basename(path))[0]),
    )
    entries = []
    element_names = set()
    stats = Counter()
    for annotation_path in annotation_paths:
        video_id = os.path.splitext(os.path.basename(annotation_path))[0]
        if args.only_video_id is not None and video_id != args.only_video_id:
            continue
        if args.limit_videos is not None and stats["videos"] >= args.limit_videos:
            break
        feature_path, times_path = find_cache_pair(
            args.cache_dir, args.cache_prefix, video_id
        )
        features = np.load(feature_path).astype(np.float32)
        times = np.load(times_path).astype(np.float32)
        if features.ndim != 2 or features.shape[1] != args.feature_dim:
            raise ValueError(f"unexpected feature shape {features.shape}: {feature_path}")
        if times.shape != (features.shape[0], 4):
            raise ValueError(f"unexpected time shape {times.shape}: {times_path}")
        with open(annotation_path, "r", encoding="utf-8") as handle:
            annotation = json.load(handle)
        if annotation_signature(annotation) in val_signatures:
            stats["excluded_validation_videos"] += 1
            continue

        valid_intervals = []
        for position, (name, element) in enumerate(
            annotation.get("executed_element", {}).items()
        ):
            try:
                start, end = parse_time(element.get("time"))
            except ValueError:
                stats["invalid_time"] += 1
                continue
            valid_intervals.append((start, end))
            element_name = canonical_element(element.get("element"))
            coarse = infer_coarse(element_name, element.get("coarse_class"))
            element_names.add(element_name)
            prototypes = select_prototypes(
                features,
                times,
                start,
                end,
                PROTOTYPE_LIMITS[coarse],
            )
            goe_grade = trimmed_judge_grade(element)
            goe_points = float(element.get("goe", float("nan")))
            bv = float(element.get("bv", float("nan")))
            panel_score = float(element.get("score_of_pannel", float("nan")))
            valid_score = all(
                np.isfinite(value) for value in (goe_grade, bv, panel_score)
            )
            for prototype_index, prototype in enumerate(prototypes):
                entries.append(
                    {
                        "key": prototype,
                        "video_id": video_id,
                        "instance_id": f"{video_id}:{name}",
                        "coarse": coarse,
                        "element": element_name,
                        "goe_grade": goe_grade if valid_score else 0.0,
                        "goe_points": goe_points if np.isfinite(goe_points) else 0.0,
                        "bv": bv if valid_score else 0.0,
                        "panel_score": panel_score if valid_score else 0.0,
                        "prototype_index": prototype_index,
                        "prototype_count": len(prototypes),
                        "start_time": start,
                        "end_time": end,
                        "valid_score": valid_score,
                    }
                )
            stats[coarse] += 1
        background_candidates = []
        for token_index, token_times in enumerate(times):
            sample_start, sample_end = float(token_times[2]), float(token_times[3])
            overlaps_action = any(
                min(sample_end, action_end) > max(sample_start, action_start)
                for action_start, action_end in valid_intervals
            )
            if not overlaps_action:
                background_candidates.append(token_index)
        if background_candidates:
            selected_positions = np.linspace(
                0,
                len(background_candidates) - 1,
                num=min(args.max_background_per_video, len(background_candidates)),
                dtype=int,
            )
            for background_index, candidate_position in enumerate(selected_positions):
                token_index = background_candidates[int(candidate_position)]
                entries.append(
                    {
                        "key": features[token_index],
                        "video_id": video_id,
                        "instance_id": f"{video_id}:background:{background_index}",
                        "coarse": "background",
                        "element": "<background>",
                        "goe_grade": 0.0,
                        "goe_points": 0.0,
                        "bv": 0.0,
                        "panel_score": 0.0,
                        "prototype_index": 0,
                        "prototype_count": 1,
                        "start_time": float(times[token_index, 2]),
                        "end_time": float(times[token_index, 3]),
                        "valid_score": False,
                    }
                )
                stats["background"] += 1
        stats["videos"] += 1

    if not entries:
        raise ValueError("corpus is empty")
    element_vocab = ["<unknown>", "<background>"] + sorted(element_names)
    element_to_id = {name: index for index, name in enumerate(element_vocab)}
    coarse_to_id = {name: index for index, name in enumerate(COARSE_CLASSES)}
    keys = F.normalize(
        torch.from_numpy(np.stack([entry["key"] for entry in entries])), dim=-1
    )
    corpus = {
        "keys": keys,
        "video_ids": [entry["video_id"] for entry in entries],
        "instance_ids": [entry["instance_id"] for entry in entries],
        "coarse_class_ids": torch.tensor(
            [coarse_to_id[entry["coarse"]] for entry in entries], dtype=torch.long
        ),
        "element_ids": torch.tensor(
            [element_to_id.get(entry["element"], 0) for entry in entries],
            dtype=torch.long,
        ),
        "elements": [entry["element"] for entry in entries],
        "goe_grades": torch.tensor(
            [entry["goe_grade"] for entry in entries], dtype=torch.float32
        ),
        "goe_points": torch.tensor(
            [entry["goe_points"] for entry in entries], dtype=torch.float32
        ),
        "bvs": torch.tensor([entry["bv"] for entry in entries], dtype=torch.float32),
        "panel_scores": torch.tensor(
            [entry["panel_score"] for entry in entries], dtype=torch.float32
        ),
        "prototype_indices": torch.tensor(
            [entry["prototype_index"] for entry in entries], dtype=torch.long
        ),
        "prototype_counts": torch.tensor(
            [entry["prototype_count"] for entry in entries], dtype=torch.long
        ),
        "start_times": torch.tensor(
            [entry["start_time"] for entry in entries], dtype=torch.float32
        ),
        "end_times": torch.tensor(
            [entry["end_time"] for entry in entries], dtype=torch.float32
        ),
        "valid_score_mask": torch.tensor(
            [entry["valid_score"] for entry in entries], dtype=torch.bool
        ),
        "coarse_class_vocab": COARSE_CLASSES,
        "element_vocab": element_vocab,
        "metadata_version": "finefs-action-rag-v1",
        "source": "FineFS annotations and DINOv2 ViT-L/14 CLS+patch-mean caches",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(corpus, args.output)
    print("saved", args.output)
    print("keys", tuple(keys.shape))
    print("action_instances", sum(stats[name] for name in PROTOTYPE_LIMITS))
    print("prototypes", len(entries))
    print("valid_score_prototypes", int(corpus["valid_score_mask"].sum()))
    print("coarse_counts", dict(stats))
    print("element_vocab", len(element_vocab))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finefs-root", default="../FineFS")
    parser.add_argument("--fs1000-root", default="../FS1000 Dataset")
    parser.add_argument(
        "--cache-dir", default="../FineFS/static_dinov2_cls_patch_mean_rag_cache"
    )
    parser.add_argument(
        "--cache-prefix", default="static_dinov2_cls_patch_mean"
    )
    parser.add_argument("--feature-dim", type=int, default=2048)
    parser.add_argument("--max-background-per-video", type=int, default=3)
    parser.add_argument("--only-video-id", default=None)
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--output", default="rag_artifacts/action_rag_corpus.pt")
    build_corpus(parser.parse_args())


if __name__ == "__main__":
    main()
