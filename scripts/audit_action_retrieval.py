import argparse
import os
import sys
from collections import Counter, defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action_rag import load_action_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="rag_artifacts/action_rag_corpus.pt")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    corpus = load_action_corpus(args.corpus)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    keys = corpus["keys"].to(device)
    element_ids = corpus["element_ids"].long()
    coarse_ids = corpus["coarse_class_ids"].long()
    valid = corpus["valid_score_mask"].bool()
    video_ids = corpus["video_ids"]
    video_vocab = {value: index for index, value in enumerate(sorted(set(video_ids)))}
    video_id_tensor = torch.tensor(
        [video_vocab[value] for value in video_ids], dtype=torch.long, device=device
    )
    coarse_vocab = corpus["coarse_class_vocab"]

    totals = Counter()
    exact_hits = Counter()
    coarse_hits = Counter()
    reciprocal_ranks = defaultdict(float)
    evaluated = 0
    for start in range(0, len(keys), args.batch_size):
        end = min(start + args.batch_size, len(keys))
        similarities = keys[start:end] @ keys.T
        same_video = video_id_tensor[start:end].unsqueeze(1).eq(
            video_id_tensor.unsqueeze(0)
        )
        similarities.masked_fill_(same_video, -torch.inf)
        for row, corpus_index in enumerate(range(start, end)):
            if not bool(valid[corpus_index]):
                continue
            if not torch.isfinite(similarities[row]).any():
                continue
            ranking = torch.topk(
                similarities[row], min(args.top_k, len(keys)), dim=-1
            ).indices.cpu()
            target_element = int(element_ids[corpus_index])
            target_coarse = int(coarse_ids[corpus_index])
            coarse_name = coarse_vocab[target_coarse]
            exact = element_ids[ranking].eq(target_element)
            coarse = coarse_ids[ranking].eq(target_coarse)
            totals[coarse_name] += 1
            exact_hits[coarse_name] += int(exact.any())
            coarse_hits[coarse_name] += int(coarse.any())
            if exact.any():
                first_rank = int(torch.nonzero(exact, as_tuple=False)[0]) + 1
                reciprocal_ranks[coarse_name] += 1.0 / first_rank
            evaluated += 1

    if evaluated == 0:
        raise ValueError("no valid scored prototypes to audit")
    print("device", device)
    print("evaluated_prototypes", evaluated)
    for name in sorted(totals):
        total = totals[name]
        print(
            name,
            "count", total,
            "exact_recall@{}".format(args.top_k), exact_hits[name] / total,
            "coarse_recall@{}".format(args.top_k), coarse_hits[name] / total,
            "exact_mrr@{}".format(args.top_k), reciprocal_ranks[name] / total,
        )
    total_exact = sum(exact_hits.values()) / evaluated
    total_coarse = sum(coarse_hits.values()) / evaluated
    total_mrr = sum(reciprocal_ranks.values()) / evaluated
    print("overall_exact_recall@{}".format(args.top_k), total_exact)
    print("overall_coarse_recall@{}".format(args.top_k), total_coarse)
    print("overall_exact_mrr@{}".format(args.top_k), total_mrr)


if __name__ == "__main__":
    main()
