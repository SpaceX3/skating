import argparse
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
import torch.utils.data as data

from action_rag import load_action_corpus
from semantic_data import (
    FineFSSemanticDataset,
    closed_set_element_ids,
    load_video_split,
    make_video_split,
    save_video_split,
    split_statistics,
)
from semantic_rag import FineFSSemanticRAG, multi_positive_nll, soft_target_nll


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["split", "train", "eval"], required=True)
    parser.add_argument("--corpus", default="rag_artifacts/action_rag_corpus.pt")
    parser.add_argument(
        "--split-path", default="rag_artifacts/finefs_semantic_split.json"
    )
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="rag_results/semantic")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-6)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--positive-count", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--retrieval-pool", type=int, default=64)
    parser.add_argument("--reference-batch-size", type=int, default=1024)
    parser.add_argument("--evidence-dim", type=int, default=256)
    parser.add_argument("--encoder-hidden-dim", type=int, default=512)
    parser.add_argument("--metadata-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--retrieval-loss-weight", type=float, default=1.0)
    parser.add_argument("--citation-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=0.25)
    parser.add_argument("--coarse-loss-weight", type=float, default=0.25)
    parser.add_argument("--goe-loss-weight", type=float, default=1.0)
    parser.add_argument("--relative-goe-loss-weight", type=float, default=0.5)
    parser.add_argument("--element-loss-weight", type=float, default=0.5)
    parser.add_argument("--direct-goe-loss-weight", type=float, default=0.5)
    parser.add_argument("--citation-goe-temperature", type=float, default=1.0)
    parser.add_argument("--hard-negative-refresh", type=int, default=1)
    parser.add_argument("--mining-batch-size", type=int, default=512)
    parser.add_argument("--random-negative-count", type=int, default=8)
    parser.add_argument(
        "--selection-metric",
        choices=[
            "closed_set_reranked_exact_mrr",
            "reranked_exact_mrr",
            "goe_within_1.0",
        ],
        default="closed_set_reranked_exact_mrr",
    )
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_score_mask(corpus):
    valid = corpus["valid_score_mask"].bool()
    goe = corpus["goe_grades"].float()
    finite = torch.isfinite(goe)
    corrected = valid & finite & goe.ge(-5.0) & goe.le(5.0)
    removed = int((valid & ~corrected).sum().item())
    if removed:
        corpus["valid_score_mask"] = corrected
        print("ignored invalid GOE prototypes:", removed)
    return removed


def element_goe_priors(corpus, train_video_ids):
    train_videos = set(str(value) for value in train_video_ids)
    by_element = defaultdict(list)
    seen_instances = set()
    all_values = []
    for index, video_id in enumerate(corpus["video_ids"]):
        instance_id = str(corpus["instance_ids"][index])
        if (
            str(video_id) not in train_videos
            or instance_id in seen_instances
            or not bool(corpus["valid_score_mask"][index])
        ):
            continue
        seen_instances.add(instance_id)
        element_id = int(corpus["element_ids"][index])
        value = float(corpus["goe_grades"][index])
        by_element[element_id].append(value)
        all_values.append(value)
    if not all_values:
        raise ValueError("semantic training has no valid GOE labels")
    fallback = float(np.median(all_values))
    priors = torch.full(
        (len(corpus["element_vocab"]),), fallback, dtype=torch.float32
    )
    for element_id, values in by_element.items():
        priors[element_id] = float(np.median(values))
    background_id = list(corpus["element_vocab"]).index("<background>")
    priors[background_id] = 0.0
    return priors


def model_from_args(args, corpus, device):
    return FineFSSemanticRAG(
        coarse_classes=len(corpus["coarse_class_vocab"]),
        elements=len(corpus["element_vocab"]),
        query_dim=corpus["keys"].shape[1],
        evidence_dim=args.evidence_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        metadata_dim=args.metadata_dim,
        temperature=args.temperature,
        dropout=args.dropout,
    ).to(device)


def model_from_checkpoint(checkpoint, corpus, device):
    config = checkpoint.get("config", {})
    model = FineFSSemanticRAG(
        coarse_classes=len(corpus["coarse_class_vocab"]),
        elements=len(corpus["element_vocab"]),
        query_dim=corpus["keys"].shape[1],
        evidence_dim=int(config.get("evidence_dim", 256)),
        encoder_hidden_dim=int(config.get("encoder_hidden_dim", 512)),
        metadata_dim=int(config.get("metadata_dim", 64)),
        temperature=float(config.get("temperature", 0.07)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def prepare_device_corpus(corpus, device):
    video_vocab = {
        value: index
        for index, value in enumerate(sorted(set(str(v) for v in corpus["video_ids"])))
    }
    instance_vocab = {
        value: index
        for index, value in enumerate(sorted(set(str(v) for v in corpus["instance_ids"])))
    }
    if "prototype_counts" in corpus:
        sample_weight = corpus["prototype_counts"].float().clamp_min(1.0).reciprocal()
    else:
        counts = defaultdict(int)
        for value in corpus["instance_ids"]:
            counts[str(value)] += 1
        sample_weight = torch.tensor(
            [1.0 / counts[str(value)] for value in corpus["instance_ids"]],
            dtype=torch.float32,
        )
    return {
        "features": corpus["keys"].to(device),
        "coarse": corpus["coarse_class_ids"].long().to(device),
        "element": corpus["element_ids"].long().to(device),
        "goe": corpus["goe_grades"].float().to(device),
        "bv": corpus["bvs"].float().to(device),
        "panel": corpus["panel_scores"].float().to(device),
        "score_valid": corpus["valid_score_mask"].bool().to(device),
        "video": torch.tensor(
            [video_vocab[str(value)] for value in corpus["video_ids"]],
            dtype=torch.long,
            device=device,
        ),
        "instance": torch.tensor(
            [instance_vocab[str(value)] for value in corpus["instance_ids"]],
            dtype=torch.long,
            device=device,
        ),
        "sample_weight": sample_weight.to(device),
    }


def gather(device_corpus, indices):
    indices = indices.to(device_corpus["features"].device, dtype=torch.long)
    return {
        name: values[indices]
        for name, values in device_corpus.items()
        if name != "instance"
    }


def weighted_mean(values, weights, mask=None):
    if mask is not None:
        weights = weights * mask.to(weights.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def forward_training_batch(
    model,
    device_corpus,
    query_indices,
    reference_indices,
    candidate_valid=None,
):
    query = gather(device_corpus, query_indices)
    reference = gather(device_corpus, reference_indices)
    different_video = reference["video"].ne(query["video"].unsqueeze(1))
    if candidate_valid is None:
        candidate_valid = torch.ones_like(different_video)
    else:
        candidate_valid = candidate_valid.to(
            different_video.device, dtype=torch.bool
        )
    positive_mask = (
        reference["element"].eq(query["element"].unsqueeze(1))
        & different_video
        & candidate_valid
    )
    candidate_similarities = F.cosine_similarity(
        query["features"].unsqueeze(1), reference["features"], dim=-1
    )
    output = model(
        query["features"],
        reference["features"],
        reference["coarse"],
        reference["element"],
        reference["goe"],
        reference["bv"],
        reference["panel"],
        reference["score_valid"],
        candidate_valid_mask=candidate_valid,
        candidate_similarities=candidate_similarities,
    )
    return output, query, reference, positive_mask, candidate_valid


def semantic_losses(args, corpus, output, query, reference, positive_mask, candidate_valid):
    weights = query["sample_weight"]
    retrieval_loss, retrieval_rows = multi_positive_nll(
        output["retrieval_logits"], positive_mask, candidate_valid, weights
    )
    goe_distance = (query["goe"].unsqueeze(1) - reference["goe"]).abs()
    scored_pair = query["score_valid"].unsqueeze(1) & reference["score_valid"]
    quality_targets = torch.exp(
        -goe_distance / max(float(args.citation_goe_temperature), 1e-6)
    )
    citation_targets = positive_mask.to(quality_targets.dtype)
    citation_targets = torch.where(
        scored_pair, citation_targets * quality_targets, citation_targets
    )
    citation_loss, citation_rows = soft_target_nll(
        output["citation_logits"], citation_targets, candidate_valid, weights
    )
    background_id = list(corpus["coarse_class_vocab"]).index("background")
    is_action = query["coarse"].ne(background_id)
    action_per_sample = F.binary_cross_entropy_with_logits(
        output["action_logit"], is_action.to(output["action_logit"].dtype), reduction="none"
    )
    action_counts = torch.bincount(is_action.long(), minlength=2).float().clamp_min(1.0)
    action_class_weights = is_action.numel() / (2.0 * action_counts)
    action_loss = weighted_mean(
        action_per_sample, weights * action_class_weights[is_action.long()]
    )
    coarse_counts = torch.bincount(
        query["coarse"], minlength=len(corpus["coarse_class_vocab"])
    ).float().clamp_min(1.0)
    coarse_class_weights = query["coarse"].numel() / (
        len(corpus["coarse_class_vocab"]) * coarse_counts
    )
    coarse_per_sample = F.cross_entropy(
        output["coarse_logits"],
        query["coarse"],
        weight=coarse_class_weights,
        reduction="none",
    )
    coarse_loss = weighted_mean(coarse_per_sample, weights)

    element_per_sample = F.cross_entropy(
        output["element_logits"], query["element"], reduction="none"
    )
    element_loss = weighted_mean(element_per_sample, weights)

    goe_per_sample = F.smooth_l1_loss(
        output["predicted_goe"], query["goe"], reduction="none"
    )
    goe_mask = is_action & query["score_valid"] & output["goe_confidence"].gt(0)
    goe_loss = weighted_mean(goe_per_sample, weights, goe_mask)
    direct_goe_per_sample = F.smooth_l1_loss(
        output["direct_goe"], query["goe"], reduction="none"
    )
    direct_goe_loss = weighted_mean(
        direct_goe_per_sample, weights, is_action & query["score_valid"]
    )

    relative_target = query["goe"].unsqueeze(1) - reference["goe"]
    relative_pair = F.smooth_l1_loss(
        output["relative_goe"], relative_target, reduction="none"
    )
    relative_mask = (
        positive_mask
        & query["score_valid"].unsqueeze(1)
        & reference["score_valid"]
    )
    relative_per_sample = (
        relative_pair * relative_mask.to(relative_pair.dtype)
    ).sum(dim=1) / relative_mask.sum(dim=1).clamp_min(1).to(relative_pair.dtype)
    relative_loss = weighted_mean(
        relative_per_sample, weights, relative_mask.any(dim=1)
    )

    total = (
        args.retrieval_loss_weight * retrieval_loss
        + args.citation_loss_weight * citation_loss
        + args.action_loss_weight * action_loss
        + args.coarse_loss_weight * coarse_loss
        + args.element_loss_weight * element_loss
        + args.goe_loss_weight * goe_loss
        + args.direct_goe_loss_weight * direct_goe_loss
        + args.relative_goe_loss_weight * relative_loss
    )
    return {
        "total": total,
        "retrieval": retrieval_loss,
        "citation": citation_loss,
        "action": action_loss,
        "coarse": coarse_loss,
        "element": element_loss,
        "goe": goe_loss,
        "direct_goe": direct_goe_loss,
        "relative_goe": relative_loss,
        "retrieval_supervised_ratio": retrieval_rows.float().mean(),
        "citation_supervised_ratio": citation_rows.float().mean(),
        "goe_supervised_ratio": goe_mask.float().mean(),
    }


def mine_training_candidates(
    args, model, device_corpus, query_indices, reference_indices
):
    """Mine exact positives and current-model hard negatives against the full bank."""
    if args.positive_count >= args.candidate_count:
        raise ValueError("positive_count must be smaller than candidate_count")
    if args.random_negative_count < 0:
        raise ValueError("random_negative_count cannot be negative")
    device = device_corpus["features"].device
    model.eval()
    bank_indices = torch.as_tensor(reference_indices, dtype=torch.long, device=device)
    bank_embeddings = encode_reference_bank(
        model, device_corpus, reference_indices, args.reference_batch_size
    )
    bank_element = device_corpus["element"][bank_indices]
    bank_video = device_corpus["video"][bank_indices]
    table_indices = torch.full(
        (device_corpus["features"].shape[0], args.candidate_count),
        int(bank_indices[0]),
        dtype=torch.long,
        device=device,
    )
    table_valid = torch.zeros_like(table_indices, dtype=torch.bool)
    hard_count = (
        args.candidate_count
        - args.positive_count
        - args.random_negative_count
    )
    if hard_count <= 0:
        raise ValueError("candidate counts leave no room for hard negatives")
    with torch.no_grad():
        for start in range(0, len(query_indices), args.mining_batch_size):
            query_batch = torch.as_tensor(
                query_indices[start : start + args.mining_batch_size],
                dtype=torch.long,
                device=device,
            )
            query = gather(device_corpus, query_batch)
            query_token, query_embeddings = model.encode_query(query["features"])
            element_logits = model.element_head(query_token)
            similarities = model.retrieval_scores(
                query_embeddings,
                element_logits,
                bank_embeddings,
                bank_element,
            )
            cross_video = query["video"].unsqueeze(1).ne(bank_video.unsqueeze(0))
            same_element = query["element"].unsqueeze(1).eq(
                bank_element.unsqueeze(0)
            )
            positive_scores, positive_positions = similarities.masked_fill(
                ~(cross_video & same_element), -torch.inf
            ).topk(args.positive_count, dim=1)
            hard_scores, hard_positions = similarities.masked_fill(
                ~(cross_video & ~same_element), -torch.inf
            ).topk(hard_count, dim=1)
            positive_valid = torch.isfinite(positive_scores)
            hard_valid = torch.isfinite(hard_scores)
            random_positions = torch.randint(
                0,
                len(reference_indices),
                (
                    query_batch.shape[0],
                    args.random_negative_count,
                ),
                device=device,
            )
            random_valid = (
                query["video"].unsqueeze(1).ne(bank_video[random_positions])
                & query["element"].unsqueeze(1).ne(
                    bank_element[random_positions]
                )
            )
            selected = torch.cat(
                [
                    bank_indices[positive_positions],
                    bank_indices[hard_positions],
                    bank_indices[random_positions],
                ],
                dim=1,
            )
            valid = torch.cat(
                [positive_valid, hard_valid, random_valid], dim=1
            )
            selected = selected.masked_fill(~valid, int(bank_indices[0]))
            table_indices[query_batch] = selected
            table_valid[query_batch] = valid
    return table_indices, table_valid


def train_epoch(
    args,
    loader,
    candidate_indices,
    candidate_valid,
    model,
    corpus,
    device_corpus,
    optimizer,
):
    model.train()
    totals = {}
    samples = 0
    for batch_index, query_indices in enumerate(loader):
        if args.limit_train_batches is not None and batch_index >= args.limit_train_batches:
            break
        device_queries = query_indices.to(device_corpus["features"].device)
        reference_indices = candidate_indices[device_queries]
        batch_candidate_valid = candidate_valid[device_queries]
        optimizer.zero_grad(set_to_none=True)
        (
            output,
            query,
            reference,
            positive_mask,
            effective_candidate_valid,
        ) = forward_training_batch(
            model,
            device_corpus,
            query_indices,
            reference_indices,
            batch_candidate_valid,
        )
        losses = semantic_losses(
            args,
            corpus,
            output,
            query,
            reference,
            positive_mask,
            effective_candidate_valid,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        batch_size = len(query_indices)
        samples += batch_size
        for name, value in losses.items():
            contribution = value.detach() * batch_size
            totals[name] = contribution if name not in totals else totals[name] + contribution
    if samples == 0:
        raise ValueError("semantic training processed zero samples")
    names = list(totals)
    values = torch.stack([totals[name] / samples for name in names]).cpu().tolist()
    result = dict(zip(names, values))
    result["samples"] = samples
    return result


def encode_reference_bank(model, device_corpus, reference_indices, batch_size):
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(reference_indices), batch_size):
            indices = torch.tensor(reference_indices[start : start + batch_size])
            reference = gather(device_corpus, indices)
            _, normalized, _ = model.encode_reference(
                reference["features"],
                reference["coarse"],
                reference["element"],
                reference["goe"],
                reference["bv"],
                reference["panel"],
            )
            embeddings.append(normalized)
    return torch.cat(embeddings, dim=0)


def deduplicated_indices(top_indices, bank_indices, bank_instance_ids, top_k):
    """Keep the first prototype per action instance entirely on the device."""
    pool_size = top_indices.shape[1]
    ranked_instances = bank_instance_ids[top_indices]
    same_instance = ranked_instances.unsqueeze(2).eq(ranked_instances.unsqueeze(1))
    has_previous = (
        same_instance
        & torch.tril(
            torch.ones(pool_size, pool_size, dtype=torch.bool, device=top_indices.device),
            diagonal=-1,
        ).unsqueeze(0)
    ).any(dim=2)
    positions = torch.arange(pool_size, device=top_indices.device).unsqueeze(0)
    positions = positions.expand_as(top_indices).masked_fill(has_previous, pool_size)
    chosen_positions = positions.topk(top_k, dim=1, largest=False).values
    if bool(chosen_positions.ge(pool_size).any()):
        raise ValueError("retrieval pool is too small after instance deduplication")
    selected_bank_positions = top_indices.gather(1, chosen_positions)
    return bank_indices[selected_bank_positions]


def rank_metrics(records, prefix):
    if not records:
        return {"count": 0}
    weights = np.asarray([record.get("weight", 1.0) for record in records])
    ranks = []
    coarse_ranks = []
    for record in records:
        exact = record[prefix + "_exact"]
        coarse = record[prefix + "_coarse"]
        ranks.append(next((i + 1 for i, value in enumerate(exact) if value), None))
        coarse_ranks.append(next((i + 1 for i, value in enumerate(coarse) if value), None))
    top_k = len(records[0][prefix + "_exact"])
    return {
        "count": len(records),
        "action_instance_weight": float(weights.sum()),
        "exact_top1": float(np.average([rank == 1 for rank in ranks], weights=weights)),
        "exact_recall_at_k": float(
            np.average([rank is not None for rank in ranks], weights=weights)
        ),
        "exact_mrr": float(
            np.average(
                [0.0 if rank is None else 1.0 / rank for rank in ranks],
                weights=weights,
            )
        ),
        "coarse_top1": float(
            np.average([rank == 1 for rank in coarse_ranks], weights=weights)
        ),
        "coarse_recall_at_k": float(
            np.average(
                [rank is not None and rank <= top_k for rank in coarse_ranks],
                weights=weights,
            )
        ),
    }


def goe_metrics(truth, prediction):
    if not truth:
        return {"count": 0}
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    absolute = np.abs(prediction - truth)
    correlation = spearmanr(truth, prediction).correlation
    return {
        "count": len(truth),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "spearman": float(correlation),
        "within_0.5": float(np.mean(absolute <= 0.5)),
        "within_1.0": float(np.mean(absolute <= 1.0)),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
    }


def evaluate(args, model, corpus, manifest, split_name, device, device_corpus=None):
    model.eval()
    if device_corpus is None:
        device_corpus = prepare_device_corpus(corpus, device)
    train_videos = set(manifest["splits"]["train"])
    split_videos = manifest["splits"][split_name]
    dataset = FineFSSemanticDataset(corpus, split_videos)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    background_id = list(corpus["coarse_class_vocab"]).index("background")
    reference_indices = [
        index
        for index, video_id in enumerate(corpus["video_ids"])
        if str(video_id) in train_videos
        and (
            bool(corpus["valid_score_mask"][index])
            or int(corpus["coarse_class_ids"][index]) == background_id
        )
    ]
    bank_indices = torch.tensor(reference_indices, dtype=torch.long, device=device)
    bank_embeddings = encode_reference_bank(
        model, device_corpus, reference_indices, args.reference_batch_size
    )
    bank_video = device_corpus["video"][bank_indices]
    bank_instance = device_corpus["instance"][bank_indices]
    closed_elements = closed_set_element_ids(corpus, train_videos)
    prototype_counts = corpus.get("prototype_counts")
    records = []
    action_truth = []
    action_prediction = []
    coarse_truth = []
    coarse_prediction = []
    element_truth = []
    element_prediction = []
    query_weights = []
    goe_by_instance = defaultdict(
        lambda: {
            "truth": None,
            "prediction": [],
            "direct": [],
            "evidence": [],
            "gate": [],
        }
    )
    query_count = 0
    with torch.no_grad():
        for batch_index, query_indices in enumerate(loader):
            if args.limit_eval_batches is not None and batch_index >= args.limit_eval_batches:
                break
            query = gather(device_corpus, query_indices)
            query_token, query_embeddings = model.encode_query(query["features"])
            element_logits = model.element_head(query_token)
            similarities = model.retrieval_scores(
                query_embeddings,
                element_logits,
                bank_embeddings,
                device_corpus["element"][bank_indices],
            )
            if split_name == "train":
                similarities.masked_fill_(
                    query["video"].unsqueeze(1).eq(bank_video.unsqueeze(0)), -torch.inf
                )
            pool_size = min(max(args.retrieval_pool, args.top_k), similarities.shape[1])
            top = torch.topk(similarities, pool_size, dim=-1).indices
            selected = deduplicated_indices(
                top, bank_indices, bank_instance, args.top_k
            )
            reference = gather(device_corpus, selected)
            dino_similarity = F.cosine_similarity(
                query["features"].unsqueeze(1), reference["features"], dim=-1
            )
            output = model(
                query["features"],
                reference["features"],
                reference["coarse"],
                reference["element"],
                reference["goe"],
                reference["bv"],
                reference["panel"],
                reference["score_valid"],
                candidate_similarities=dino_similarity,
            )
            rerank = output["citation_weights"].argsort(dim=-1, descending=True)
            predicted_action_batch = torch.sigmoid(output["action_logit"]).ge(0.5).cpu().tolist()
            predicted_coarse_batch = output["coarse_logits"].argmax(dim=-1).cpu().tolist()
            predicted_element_batch = output["element_logits"].argmax(dim=-1).cpu().tolist()
            raw_element_batch = reference["element"].cpu().tolist()
            raw_coarse_batch = reference["coarse"].cpu().tolist()
            rerank_batch = rerank.cpu().tolist()
            predicted_goe_batch = output["predicted_goe"].cpu().tolist()
            direct_goe_batch = output["direct_goe"].cpu().tolist()
            evidence_goe_batch = output["evidence_goe"].cpu().tolist()
            goe_gate_batch = output["goe_gate"].cpu().tolist()
            for row, query_index in enumerate(query_indices.tolist()):
                query_count += 1
                target_coarse = int(corpus["coarse_class_ids"][query_index])
                target_element = int(corpus["element_ids"][query_index])
                is_action = target_coarse != background_id
                prototype_count = (
                    1 if prototype_counts is None else int(prototype_counts[query_index])
                )
                query_weight = 1.0 / max(prototype_count, 1)
                predicted_action = predicted_action_batch[row]
                action_truth.append(is_action)
                action_prediction.append(predicted_action)
                coarse_truth.append(target_coarse)
                coarse_prediction.append(predicted_coarse_batch[row])
                element_truth.append(target_element)
                element_prediction.append(predicted_element_batch[row])
                query_weights.append(query_weight)
                if not is_action:
                    continue
                raw_element = raw_element_batch[row]
                raw_coarse = raw_coarse_batch[row]
                order = rerank_batch[row]
                record = {
                    "closed_set": target_element in closed_elements,
                    "coarse_name": corpus["coarse_class_vocab"][target_coarse],
                    "weight": query_weight,
                    "raw_exact": [value == target_element for value in raw_element],
                    "raw_coarse": [value == target_coarse for value in raw_coarse],
                    "reranked_exact": [
                        raw_element[position] == target_element for position in order
                    ],
                    "reranked_coarse": [
                        raw_coarse[position] == target_coarse for position in order
                    ],
                }
                records.append(record)
                if bool(corpus["valid_score_mask"][query_index]):
                    instance_id = corpus["instance_ids"][query_index]
                    goe_by_instance[instance_id]["truth"] = float(
                        corpus["goe_grades"][query_index]
                    )
                    goe_by_instance[instance_id]["prediction"].append(
                        predicted_goe_batch[row]
                    )
                    goe_by_instance[instance_id]["direct"].append(
                        direct_goe_batch[row]
                    )
                    goe_by_instance[instance_id]["evidence"].append(
                        evidence_goe_batch[row]
                    )
                    goe_by_instance[instance_id]["gate"].append(
                        goe_gate_batch[row]
                    )
    closed_records = [record for record in records if record["closed_set"]]
    goe_truth = []
    goe_prediction = []
    direct_goe_prediction = []
    evidence_goe_prediction = []
    goe_gates = []
    for item in goe_by_instance.values():
        goe_truth.append(item["truth"])
        goe_prediction.append(float(np.mean(item["prediction"])))
        direct_goe_prediction.append(float(np.mean(item["direct"])))
        evidence_goe_prediction.append(float(np.mean(item["evidence"])))
        goe_gates.append(float(np.mean(item["gate"])))
    per_coarse = {}
    for name in sorted(set(record["coarse_name"] for record in records)):
        subset = [record for record in records if record["coarse_name"] == name]
        per_coarse[name] = {
            "raw": rank_metrics(subset, "raw"),
            "reranked": rank_metrics(subset, "reranked"),
        }
    return {
        "split": split_name,
        "queries": query_count,
        "action_queries": len(records),
        "closed_set_action_queries": len(closed_records),
        "open_set_action_queries": len(records) - len(closed_records),
        "top_k": args.top_k,
        "action_accuracy": float(
            np.average(
                np.asarray(action_truth) == np.asarray(action_prediction),
                weights=np.asarray(query_weights),
            )
        ),
        "coarse_accuracy": float(
            np.average(
                np.asarray(coarse_truth) == np.asarray(coarse_prediction),
                weights=np.asarray(query_weights),
            )
        ),
        "element_accuracy": float(
            np.average(
                np.asarray(element_truth) == np.asarray(element_prediction),
                weights=np.asarray(query_weights),
            )
        ),
        "raw": rank_metrics(records, "raw"),
        "reranked": rank_metrics(records, "reranked"),
        "closed_set_raw": rank_metrics(closed_records, "raw"),
        "closed_set_reranked": rank_metrics(closed_records, "reranked"),
        "per_coarse": per_coarse,
        "goe_action_instance": goe_metrics(goe_truth, goe_prediction),
        "direct_goe_action_instance": goe_metrics(
            goe_truth, direct_goe_prediction
        ),
        "evidence_goe_action_instance": goe_metrics(
            goe_truth, evidence_goe_prediction
        ),
        "goe_gate_mean": float(np.mean(goe_gates)),
    }


def save_checkpoint(path, model, optimizer, args, corpus, manifest, epoch, validation):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_stage": "finefs_semantic",
            "epoch": epoch,
            "validation": validation,
            "config": vars(args),
            "corpus_version": corpus["metadata_version"],
            "split_format_version": manifest["format_version"],
        },
        path,
    )


def selection_value(validation, name):
    if name == "closed_set_reranked_exact_mrr":
        return float(validation["closed_set_reranked"]["exact_mrr"])
    if name == "reranked_exact_mrr":
        return float(validation["reranked"]["exact_mrr"])
    if name == "goe_within_1.0":
        return float(validation["goe_action_instance"]["within_1.0"])
    raise ValueError("unsupported selection metric: {}".format(name))


def train(args, corpus, manifest, device):
    model = model_from_args(args, corpus, device)
    model.set_element_goe_prior(
        element_goe_priors(corpus, manifest["splits"]["train"])
    )
    device_corpus = prepare_device_corpus(corpus, device)
    dataset = FineFSSemanticDataset(corpus, manifest["splits"]["train"])
    generator = torch.Generator().manual_seed(args.seed)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    reference_indices = list(dataset.indices)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = args.run_name or "semantic_seed{}_{}".format(
        args.seed, datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    log_path = os.path.join(args.output_dir, run_name + ".jsonl")
    best_value = -float("inf")
    best_path = None
    best_mae = float("inf")
    best_mae_path = None
    candidate_indices = None
    candidate_valid = None
    print("device:", device)
    print("train queries:", len(dataset))
    print("train reference prototypes:", len(reference_indices))
    print("log:", log_path)
    with open(log_path, "w", encoding="utf-8") as handle:
        for epoch in range(args.epochs):
            if (
                candidate_indices is None
                or epoch % max(args.hard_negative_refresh, 1) == 0
            ):
                mining_start = time.perf_counter()
                candidate_indices, candidate_valid = mine_training_candidates(
                    args,
                    model,
                    device_corpus,
                    dataset.indices,
                    reference_indices,
                )
                print(
                    "mined full-bank candidates for epoch {} in {:.2f}s".format(
                        epoch + 1, time.perf_counter() - mining_start
                    )
                )
            train_result = train_epoch(
                args,
                loader,
                candidate_indices,
                candidate_valid,
                model,
                corpus,
                device_corpus,
                optimizer,
            )
            record = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": train_result,
            }
            if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
                validation = evaluate(
                    args, model, corpus, manifest, "val", device, device_corpus
                )
                record["validation"] = validation
                value = selection_value(validation, args.selection_metric)
                if np.isfinite(value) and value > best_value:
                    best_value = value
                    goe_mae = validation["goe_action_instance"].get("mae", float("nan"))
                    best_path = os.path.join(
                        args.output_dir,
                        "{}_best_epoch{:03d}_mrr{:.4f}_goemae{:.4f}.pth".format(
                            run_name, epoch + 1, value, goe_mae
                        ),
                    )
                    save_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        args,
                        corpus,
                        manifest,
                        epoch,
                        validation,
                    )
                    print("saved best checkpoint:", best_path)
                goe_mae = validation["goe_action_instance"].get(
                    "mae", float("nan")
                )
                if np.isfinite(goe_mae) and goe_mae < best_mae:
                    best_mae = goe_mae
                    best_mae_path = os.path.join(
                        args.output_dir,
                        "{}_bestmae_epoch{:03d}_mae{:.4f}_mrr{:.4f}.pth".format(
                            run_name,
                            epoch + 1,
                            goe_mae,
                            validation["closed_set_reranked"]["exact_mrr"],
                        ),
                    )
                    save_checkpoint(
                        best_mae_path,
                        model,
                        optimizer,
                        args,
                        corpus,
                        manifest,
                        epoch,
                        validation,
                    )
                    print("saved best-MAE checkpoint:", best_mae_path)
            print(json.dumps(record, ensure_ascii=False))
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            scheduler.step()
    print("best validation {}: {}".format(args.selection_metric, best_value))
    print("best checkpoint:", best_path)
    print("best validation GOE MAE:", best_mae)
    print("best-MAE checkpoint:", best_mae_path)


def main():
    args = parse_args()
    set_seed(args.seed)
    corpus = load_action_corpus(args.corpus)
    sanitize_score_mask(corpus)
    if args.mode == "split":
        manifest = make_video_split(
            corpus,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        save_video_split(args.split_path, manifest, overwrite=args.overwrite_split)
        print("saved split:", args.split_path)
        print(json.dumps(split_statistics(corpus, manifest), ensure_ascii=False, indent=2))
        return
    manifest = load_video_split(args.split_path, corpus)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda:{}".format(args.gpu)
    )
    if args.mode == "train":
        train(args, corpus, manifest, device)
        return
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for semantic evaluation")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint.get("training_stage") != "finefs_semantic":
        raise ValueError("checkpoint is not a FineFS semantic checkpoint")
    if checkpoint.get("corpus_version") != corpus["metadata_version"]:
        raise ValueError("checkpoint and action corpus versions differ")
    model = model_from_checkpoint(checkpoint, corpus, device)
    result = evaluate(args, model, corpus, manifest, args.eval_split, device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
