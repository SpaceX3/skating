import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.utils.data as data

from action_rag import load_action_corpus
from dataset.dataset_fs800 import (
    FeatureDatasetWithStaticCache,
    av_collate_fn_with_static,
)
from model import scoring_head


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root-path", default="../FS1000 Dataset")
    parser.add_argument("--rag-corpus-path", default="rag_artifacts/action_rag_corpus.pt")
    parser.add_argument("--candidate-dir", default="rag_artifacts/candidates")
    parser.add_argument(
        "--static-cache-dir-name", default="static_dinov2_cls_patch_mean_cache"
    )
    parser.add_argument(
        "--static-cache-prefix", default="static_dinov2_cls_patch_mean"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--force-baseline-only",
        "--force-dynamic-only",
        dest="force_baseline_only",
        action="store_true",
    )
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--top-citations", type=int, default=5)
    parser.add_argument("--only-data-index", default=None)
    return parser.parse_args()


def strip_module_prefix(state_dict):
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def metrics(truth, prediction):
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    return {
        "mse": float(np.mean((prediction - truth) ** 2)),
        "mae": float(np.mean(np.abs(prediction - truth))),
        "spearman": float(spearmanr(truth, prediction).correlation),
    }


def citations_for_sample(model, indices, weights, static_mask, top_n):
    contribution = defaultdict(float)
    for time_index in range(indices.shape[0]):
        if not bool(static_mask[time_index]):
            continue
        for index, weight in zip(indices[time_index], weights[time_index]):
            index = int(index)
            if index < 0 or not bool(model.rag.corpus_valid_score[index]):
                continue
            contribution[index] += float(weight)
    ranked = sorted(contribution, key=contribution.get, reverse=True)[:top_n]
    resolved = model.rag.resolve_citations(ranked)
    for item, index in zip(resolved, ranked):
        item["aggregated_weight"] = contribution[index]
    return resolved


def main():
    args = parse_args()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = strip_module_prefix(state_dict)
    stage = checkpoint.get("training_stage", "rag" if any(k.startswith("rag.") for k in state_dict) else "dynamic")
    config = checkpoint.get("config", {})
    use_rag = stage == "rag" and not args.force_baseline_only
    corpus = load_action_corpus(args.rag_corpus_path) if use_rag else None
    if use_rag and checkpoint.get("corpus_version") not in (
        None,
        corpus["metadata_version"],
    ):
        raise ValueError("checkpoint and corpus metadata versions differ")
    model = scoring_head(
        depth=2,
        input_dim=768,
        dim=512,
        input_len=16,
        num_scores=1,
        use_static_branch=use_rag,
        use_static_baseline=True,
        static_in_dim=2048,
        static_proj_dim=int(config.get("rag_feature_dim", 256)),
        baseline_static_proj_dim=int(
            config.get(
                "baseline_static_proj_dim",
                state_dict["static_proj.weight"].shape[0],
            )
        ),
        rag_corpus=corpus,
        rag_delta_max=float(config.get("rag_delta_max", 20.0)),
    ).to(device)
    if args.force_baseline_only and stage == "rag":
        state_dict = {key: value for key, value in state_dict.items() if not key.startswith("rag.")}
        model.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    model.eval()

    dataset = FeatureDatasetWithStaticCache(
        root_path=args.root_path,
        is_train=False,
        cache_dir_name=args.static_cache_dir_name,
        cache_prefix=args.static_cache_prefix,
        candidate_dir=args.candidate_dir if use_rag else None,
        require_candidates=use_rag,
        only_data_index=args.only_data_index,
    )
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=av_collate_fn_with_static,
        pin_memory=device.type == "cuda",
    )
    output_path = args.output_jsonl or os.path.splitext(args.checkpoint)[0] + "_predictions.jsonl"
    truth, final_prediction, baseline_prediction = [], [], []
    with open(output_path, "w", encoding="utf-8") as output_file, torch.no_grad():
        for batch in loader:
            (
                audio,
                video,
                inv_audio,
                inv_video,
                static,
                static_valid_mask,
                audio_len,
                video_len,
                scores,
                data_indices,
                rag_data,
            ) = batch
            kwargs = {}
            if rag_data is not None:
                kwargs = {
                    "candidate_indices": rag_data["candidate_indices"].to(device),
                    "candidate_similarities": rag_data[
                        "candidate_similarities"
                    ].to(device),
                    "overlap_weights": rag_data["overlap_weights"].to(device),
                }
            output = model(
                audio.to(device),
                video.to(device),
                inv_audio.to(device),
                inv_video.to(device),
                audio_len,
                video_len,
                static_feature=static.to(device),
                static_valid_mask=static_valid_mask.to(device),
                **kwargs,
            )
            target = scores[0]
            truth.extend(target.tolist())
            final_prediction.extend(output["score"].cpu().tolist())
            baseline_prediction.extend(output["tes_baseline"].cpu().tolist())
            for row, data_index in enumerate(data_indices):
                record = {
                    "data_index": data_index,
                    "target_tes": float(target[row]),
                    "tes_final": float(output["score"][row].cpu()),
                    "tes_baseline": float(output["tes_baseline"][row].cpu()),
                    "delta_tes_rag": float(output["delta_tes_rag"][row].cpu()),
                }
                if use_rag:
                    record["top_citations"] = citations_for_sample(
                        model,
                        output["citation_indices"][row].cpu(),
                        output["citation_weights"][row].cpu(),
                        static_valid_mask[row],
                        args.top_citations,
                    )
                    record["retrieval_statistics"] = output[
                        "retrieval_statistics"
                    ][row].cpu().tolist()
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    result = {
        "checkpoint": args.checkpoint,
        "stage": stage,
        "force_baseline_only": args.force_baseline_only,
        "samples": len(dataset),
        "final": metrics(truth, final_prediction),
        "baseline": metrics(truth, baseline_prediction),
        "predictions": output_path,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
