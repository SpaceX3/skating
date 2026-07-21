import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action_rag import load_action_corpus


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


def build_exact_video_map(finefs_root, fs1000_root):
    by_signature = defaultdict(list)
    for path in glob.glob(os.path.join(finefs_root, "annotation", "*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            annotation = json.load(handle)
        video_id = os.path.splitext(os.path.basename(path))[0]
        by_signature[annotation_signature(annotation)].append(video_id)

    mapping = {}
    ambiguous = []
    for split in ("train", "val"):
        split_path = os.path.join(fs1000_root, f"{split}_fs800.txt")
        with open(split_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                matches = by_signature.get(rounded(parts[1:9]), [])
                if len(matches) == 1:
                    mapping[parts[0]] = matches[0]
                elif len(matches) > 1:
                    ambiguous.append((parts[0], matches))
    if ambiguous:
        raise ValueError("ambiguous FS1000/FineFS score signatures: {}".format(ambiguous[:5]))
    return mapping


def read_split(root, split):
    path = os.path.join(root, f"{split}_fs800.txt")
    with open(path, "r", encoding="utf-8") as handle:
        return [line.split()[0] for line in handle if line.strip()]


def query_cache_path(cache_dir, prefix, data_index):
    paths = [
        path
        for path in glob.glob(os.path.join(cache_dir, f"{prefix}_{data_index}_T*.npy"))
        if not path.endswith(".times.npy")
    ]
    if len(paths) != 1:
        raise ValueError(f"expected one DINO cache for {data_index}, got {paths}")
    return paths[0]


def deduplicated_topk(similarities, corpus, topk, pool_size):
    pool_size = min(pool_size, similarities.shape[1])
    values, indices = torch.topk(similarities, pool_size, dim=-1)
    output_indices = torch.full(
        (similarities.shape[0], topk), -1, dtype=torch.long
    )
    output_values = torch.zeros(
        (similarities.shape[0], topk), dtype=torch.float32
    )
    instance_ids = corpus["instance_ids"]
    for row in range(similarities.shape[0]):
        seen = set()
        write_index = 0
        for value, index in zip(values[row].tolist(), indices[row].tolist()):
            if not np.isfinite(value):
                continue
            instance_id = instance_ids[index]
            if instance_id in seen:
                continue
            seen.add(instance_id)
            output_indices[row, write_index] = index
            output_values[row, write_index] = value
            write_index += 1
            if write_index == topk:
                break
    return output_indices, output_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs1000-root", default="../FS1000 Dataset")
    parser.add_argument("--finefs-root", default="../FineFS")
    parser.add_argument(
        "--query-cache-dir",
        default="../FS1000 Dataset/static_dinov2_cls_patch_mean_cache",
    )
    parser.add_argument(
        "--query-cache-prefix", default="static_dinov2_cls_patch_mean"
    )
    parser.add_argument("--corpus", default="rag_artifacts/action_rag_corpus.pt")
    parser.add_argument("--output-dir", default="rag_artifacts/candidates")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dedup-pool-size", type=int, default=64)
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only-data-index", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.top_k <= 0 or args.dedup_pool_size < args.top_k:
        raise ValueError("require 0 < top-k <= dedup-pool-size")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    corpus = load_action_corpus(args.corpus)
    corpus_keys = corpus["keys"].to(device)
    exact_video_map = build_exact_video_map(args.finefs_root, args.fs1000_root)
    corpus_video_ids = np.asarray(corpus["video_ids"], dtype=object)
    os.makedirs(args.output_dir, exist_ok=True)
    print("device", device)
    print("corpus_keys", tuple(corpus_keys.shape))
    print("exact_video_exclusions", len(exact_video_map))

    splits = ("train", "val") if args.split == "all" else (args.split,)
    total = 0
    for split in splits:
        data_indices = read_split(args.fs1000_root, split)
        for position, data_index in enumerate(data_indices):
            if args.only_data_index is not None and data_index != args.only_data_index:
                continue
            if args.limit is not None and total >= args.limit:
                break
            output_path = os.path.join(args.output_dir, data_index + ".npz")
            if os.path.exists(output_path) and not args.overwrite:
                continue
            feature_path = query_cache_path(
                args.query_cache_dir, args.query_cache_prefix, data_index
            )
            query = torch.from_numpy(np.load(feature_path).astype(np.float32))
            query = F.normalize(query, dim=-1).to(device)
            similarities = query @ corpus_keys.T
            excluded_video_id = exact_video_map.get(data_index)
            excluded_count = 0
            if excluded_video_id is not None:
                excluded = torch.from_numpy(
                    corpus_video_ids == excluded_video_id
                ).to(device=device, dtype=torch.bool)
                excluded_count = int(excluded.sum().item())
                similarities[:, excluded] = -torch.inf
            indices, values = deduplicated_topk(
                similarities.detach().cpu(),
                corpus,
                args.top_k,
                args.dedup_pool_size,
            )
            if (indices < 0).all(dim=-1).any():
                raise ValueError(f"no valid candidates for one or more queries: {data_index}")
            np.savez_compressed(
                output_path,
                candidate_indices=indices.numpy(),
                candidate_similarities=values.numpy(),
                corpus_version=np.asarray(corpus["metadata_version"]),
                source_feature=os.path.basename(feature_path),
                excluded_finefs_video=np.asarray(excluded_video_id or ""),
            )
            total += 1
            if (position + 1) % 25 == 0 or position + 1 == len(data_indices):
                print(
                    f"{split} {position + 1}/{len(data_indices)} "
                    f"{data_index} excluded={excluded_count}",
                    flush=True,
                )
    print("written_candidate_files", total)


if __name__ == "__main__":
    main()
