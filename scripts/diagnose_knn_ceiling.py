"""Oracle kNN ceiling diagnostic for FineFS exact-element recognition.

Question: is the 23% element top-1 a limitation of the MLP head, or of the
averaged DINO feature itself?

Method: use the raw 2048-d DINO prototype features directly. Build a reference
bank from FineFS-train action prototypes, query with FineFS-val action
prototypes, and predict the element by nearest-neighbour vote using ground-truth
reference labels. This needs no training, so it measures how much element
information the feature space actually carries.

The query/metric definitions mirror semantic_main.classifier_metrics so the
numbers are directly comparable with the trained classifier:
  - queries: val prototypes kept by FineFSSemanticDataset, then coarse != background
  - weighting: sample_weight = 1 / prototype_count (instance-balanced)
  - top-k: weighted fraction of queries whose true element is in the top k

Train and val videos are disjoint by construction, so no same-video exclusion is
needed for the val query set.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action_rag import load_action_corpus
from semantic_data import FineFSSemanticDataset, load_video_split
from semantic_rag import CLASSIFIER_STAGE, SemanticQueryClassifier, require_semantic_v2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="rag_artifacts/action_rag_corpus.pt")
    parser.add_argument("--split-path", default="rag_artifacts/finefs_semantic_split.json")
    parser.add_argument("--classifier-checkpoint", default=None,
                        help="optional: evaluate the trained MLP on the same query set")
    parser.add_argument("--query-split", choices=["val", "test"], default="val")
    parser.add_argument("--knn", type=int, default=20, help="neighbours for the vote")
    parser.add_argument("--pool", type=int, default=64,
                        help="prototype pool size for the routing recall ceiling")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def select_indices(corpus, video_ids, action_only=True):
    """Prototypes kept by FineFSSemanticDataset, optionally action-only."""
    dataset = FineFSSemanticDataset(corpus, video_ids)
    background = list(corpus["coarse_class_vocab"]).index("background")
    background_element = list(corpus["element_vocab"]).index("<background>")
    indices = []
    for index in dataset.indices:
        coarse_id = int(corpus["coarse_class_ids"][index])
        element_id = int(corpus["element_ids"][index])
        if action_only and (coarse_id == background or element_id == background_element):
            continue
        indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def sample_weights(corpus, indices):
    if "prototype_counts" in corpus:
        counts = corpus["prototype_counts"][indices].float()
    else:
        from collections import Counter
        instance_counts = Counter(str(v) for v in corpus["instance_ids"])
        counts = torch.tensor(
            [instance_counts[str(corpus["instance_ids"][int(i)])] for i in indices],
            dtype=torch.float32,
        )
    return counts.clamp_min(1.0).reciprocal().numpy().astype(np.float64)


def ranked_elements_from_neighbours(neighbour_elements, neighbour_scores, num_elements, top_n=10):
    """Similarity-weighted element vote -> ranked unique element ids."""
    batch = neighbour_elements.shape[0]
    votes = torch.zeros(batch, num_elements, device=neighbour_elements.device)
    votes.scatter_add_(1, neighbour_elements, neighbour_scores)
    return votes.topk(min(top_n, num_elements), dim=-1).indices


def ranked_elements_first_hit(neighbour_elements, top_n=10):
    """Rank by best-matching prototype: dedup the neighbour element sequence."""
    batch, k = neighbour_elements.shape
    out = torch.full((batch, top_n), -1, dtype=torch.long, device=neighbour_elements.device)
    elements = neighbour_elements.cpu().numpy()
    result = np.full((batch, top_n), -1, dtype=np.int64)
    for row in range(batch):
        seen = []
        for value in elements[row]:
            if value not in seen:
                seen.append(int(value))
                if len(seen) == top_n:
                    break
        result[row, : len(seen)] = seen
    return out.new_tensor(result)


def coverage(ranked, target, weights, k):
    ranked = ranked[:, : min(k, ranked.shape[1])]
    hit = (ranked == target[:, None]).any(axis=1)
    return float(np.average(hit, weights=weights)) if len(target) else 0.0


def report_block(name, ranked, target, weights, extra=None):
    block = {
        "element_top1": coverage(ranked, target, weights, 1),
        "element_top5": coverage(ranked, target, weights, 5),
        "element_top10": coverage(ranked, target, weights, 10),
    }
    if extra:
        block.update(extra)
    return name, block


def per_coarse_top1(ranked, target, weights, coarse, coarse_vocab):
    result = {}
    for coarse_id in sorted(set(coarse.tolist())):
        rows = coarse == coarse_id
        if not rows.any():
            continue
        result[coarse_vocab[coarse_id]] = {
            "count": int(rows.sum()),
            "element_top1": coverage(ranked[rows], target[rows], weights[rows], 1),
            "element_top5": coverage(ranked[rows], target[rows], weights[rows], 5),
        }
    return result


def verify_retrieval_encoder_gradient(corpus, device):
    """Reproduce the Stage A loss and check whether retrieval_encoder gets grads."""
    model = SemanticQueryClassifier(
        len(corpus["coarse_class_vocab"]), len(corpus["element_vocab"]),
        corpus["keys"].shape[1],
    ).to(device)
    features = torch.randn(8, corpus["keys"].shape[1], device=device)
    coarse_target = torch.randint(2, len(corpus["coarse_class_vocab"]), (8,), device=device)
    element_target = torch.randint(0, len(corpus["element_vocab"]), (8,), device=device)
    output = model(features)
    background = list(corpus["coarse_class_vocab"]).index("background")
    is_action = coarse_target.ne(background)
    action_loss = F.binary_cross_entropy_with_logits(
        output["action_logit"], is_action.float()
    )
    coarse_loss = F.cross_entropy(output["coarse_logits"], coarse_target)
    element_loss = F.cross_entropy(output["element_logits"], element_target)
    total = 0.25 * action_loss + 0.25 * coarse_loss + 0.5 * element_loss
    total.backward()
    grads = {
        "retrieval_encoder.weight": model.retrieval_encoder.weight.grad,
        "element_head.weight": model.element_head.weight.grad,
    }
    return {
        name: (None if grad is None else float(grad.abs().sum()))
        for name, grad in grads.items()
    }


def main():
    args = parse_args()
    device = torch.device("cpu") if args.cpu else torch.device("cuda", args.gpu)
    corpus = load_action_corpus(args.corpus)
    manifest = load_video_split(args.split_path, corpus)
    coarse_vocab = list(corpus["coarse_class_vocab"])
    num_elements = len(corpus["element_vocab"])

    gradient_check = verify_retrieval_encoder_gradient(corpus, torch.device("cpu"))

    bank_idx = select_indices(corpus, manifest["splits"]["train"])
    query_idx = select_indices(corpus, manifest["splits"][args.query_split])

    keys = corpus["keys"].to(device)  # already L2-normalised by load_action_corpus
    bank_features = keys[torch.from_numpy(bank_idx).to(device)]
    bank_elements = corpus["element_ids"].long()[bank_idx].to(device)
    bank_coarse = corpus["coarse_class_ids"].long()[bank_idx].to(device)

    query_features = keys[torch.from_numpy(query_idx).to(device)]
    target = corpus["element_ids"].long()[query_idx].numpy()
    query_coarse = corpus["coarse_class_ids"].long()[query_idx].numpy()
    weights = sample_weights(corpus, query_idx)

    top_n = 10
    pool = min(args.pool, bank_features.shape[0])
    neighbours = max(args.knn, top_n)

    vote_ranked, nn_ranked, pool_recall, coarse_pred = [], [], [], []
    for start in range(0, query_features.shape[0], args.chunk):
        chunk = query_features[start : start + args.chunk]
        similarity = chunk @ bank_features.T
        top_pool = similarity.topk(pool, dim=-1)
        pool_elements = bank_elements[top_pool.indices]
        chunk_target = torch.from_numpy(target[start : start + chunk.shape[0]]).to(device)

        knn_elements = pool_elements[:, :neighbours]
        knn_scores = top_pool.values[:, :neighbours].clamp_min(0.0)
        vote_ranked.append(
            ranked_elements_from_neighbours(knn_elements, knn_scores, num_elements, top_n).cpu().numpy()
        )
        nn_ranked.append(ranked_elements_first_hit(pool_elements, top_n).cpu().numpy())
        pool_recall.append(
            (pool_elements == chunk_target[:, None]).cummax(dim=1).values.cpu().numpy()
        )
        coarse_pred.append(bank_coarse[top_pool.indices[:, 0]].cpu().numpy())

    vote_ranked = np.concatenate(vote_ranked)
    nn_ranked = np.concatenate(nn_ranked)
    pool_recall = np.concatenate(pool_recall)
    coarse_pred = np.concatenate(coarse_pred)

    results = {
        "config": {
            "query_split": args.query_split,
            "knn": neighbours,
            "pool": pool,
            "bank_prototypes": int(bank_idx.size),
            "query_prototypes": int(query_idx.size),
            "action_instance_weight": float(weights.sum()),
            "elements": num_elements,
        },
        "stage_a_gradient_check": gradient_check,
    }

    name, block = report_block(
        "knn_vote", vote_ranked, target, weights,
        {"coarse_top1": float(np.average(coarse_pred == query_coarse, weights=weights))},
    )
    results[name] = block
    name, block = report_block("nearest_prototype", nn_ranked, target, weights)
    results[name] = block

    results["routing_recall_ceiling"] = {
        "recall_at_8": float(np.average(pool_recall[:, min(8, pool) - 1], weights=weights)),
        "recall_at_16": float(np.average(pool_recall[:, min(16, pool) - 1], weights=weights)),
        "recall_at_{}".format(pool): float(np.average(pool_recall[:, pool - 1], weights=weights)),
    }
    results["per_coarse_knn_vote"] = per_coarse_top1(
        vote_ranked, target, weights, query_coarse, coarse_vocab
    )

    if args.classifier_checkpoint:
        checkpoint = torch.load(args.classifier_checkpoint, map_location=device)
        require_semantic_v2(checkpoint, CLASSIFIER_STAGE)
        config = dict(checkpoint.get("config", {}))
        model = SemanticQueryClassifier(
            len(coarse_vocab), num_elements, corpus["keys"].shape[1],
            int(config.get("evidence_dim", 256)),
            int(config.get("encoder_hidden_dim", 512)),
            float(config.get("dropout", 0.1)),
        ).to(device)
        model.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
        model.eval()
        mlp_ranked, mlp_coarse = [], []
        with torch.no_grad():
            for start in range(0, query_features.shape[0], args.chunk):
                output = model.classify_query(query_features[start : start + args.chunk])
                mlp_ranked.append(output["element_logits"].topk(top_n, dim=-1).indices.cpu().numpy())
                mlp_coarse.append(output["coarse_logits"].argmax(-1).cpu().numpy())
        mlp_ranked = np.concatenate(mlp_ranked)
        mlp_coarse = np.concatenate(mlp_coarse)
        name, block = report_block(
            "mlp_classifier", mlp_ranked, target, weights,
            {"coarse_top1": float(np.average(mlp_coarse == query_coarse, weights=weights))},
        )
        results[name] = block
        results["per_coarse_mlp"] = per_coarse_top1(
            mlp_ranked, target, weights, query_coarse, coarse_vocab
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
