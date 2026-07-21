import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action_rag import load_action_corpus
from dataset.dataset_fs800 import (
    FeatureDatasetWithStaticCache,
    av_collate_fn_with_static,
)
from model import scoring_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--data-index", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    dataset = FeatureDatasetWithStaticCache(
        root_path=args.root_path,
        is_train=True,
        cache_dir_name="static_dinov2_cls_patch_mean_cache",
        cache_prefix="static_dinov2_cls_patch_mean",
        only_data_index=args.data_index,
        candidate_dir=args.candidate_dir,
        require_candidates=True,
    )
    if len(dataset) != 1:
        raise ValueError(f"expected exactly one sample, got {len(dataset)}")
    batch = av_collate_fn_with_static([dataset[0]])
    (
        audio,
        video,
        inv_audio,
        inv_video,
        static,
        static_mask,
        audio_len,
        video_len,
        scores,
        data_indices,
        rag_data,
    ) = batch
    corpus = load_action_corpus(args.corpus)
    model = scoring_head(
        depth=2,
        input_dim=768,
        dim=512,
        input_len=16,
        use_static_branch=True,
        use_static_baseline=True,
        static_in_dim=2048,
        static_proj_dim=256,
        baseline_static_proj_dim=512,
        rag_corpus=corpus,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.rag.parameters():
        parameter.requires_grad = True
    model.eval()
    model.rag.train()
    optimizer = torch.optim.Adam(model.rag.parameters(), lr=1e-3)

    kwargs = dict(
        audio_feature=audio.to(device),
        video_feature=video.to(device),
        inv_audio_feature=inv_audio.to(device),
        inv_video_feature=inv_video.to(device),
        audio_len=audio_len,
        video_len=video_len,
        static_feature=static.to(device),
        static_valid_mask=static_mask.to(device),
        candidate_indices=rag_data["candidate_indices"].to(device),
        candidate_similarities=rag_data["candidate_similarities"].to(device),
        overlap_weights=rag_data["overlap_weights"].to(device),
    )
    target = scores[0].to(device)
    first = model(**kwargs)
    if not torch.allclose(first["score"], first["tes_dynamic"]):
        raise AssertionError("zero initialized RAG correction changed the baseline")
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(**kwargs)
        loss = torch.nn.functional.mse_loss(output["score"], target)
        loss.backward()
        optimizer.step()
    final = model(**kwargs)
    if not torch.isfinite(final["score"]).all():
        raise AssertionError("model produced NaN/Inf")
    if model.rag.corpus_keys.grad is not None:
        raise AssertionError("frozen corpus keys received gradients")
    if model.linear1.weight.grad is not None:
        raise AssertionError("frozen dynamic branch received gradients")
    print("data_index", data_indices)
    print("audio", tuple(audio.shape), "video", tuple(video.shape))
    print("static", tuple(static.shape), "candidates", tuple(rag_data["candidate_indices"].shape))
    print("overlap", tuple(rag_data["overlap_weights"].shape))
    print("target", float(target[0]))
    print("tes_dynamic", float(final["tes_dynamic"][0]))
    print("delta_tes_rag", float(final["delta_tes_rag"][0]))
    print("evidence_valid", bool(final["evidence_valid_mask"][0]))
    print("real data smoke test passed")


if __name__ == "__main__":
    main()
