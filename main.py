import argparse
import json
import os
import random
from datetime import datetime

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
from semantic_rag import FineFSSemanticRAG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-stage", choices=["dynamic", "rag"], required=True)
    parser.add_argument("--root-path", default="../FS1000 Dataset")
    parser.add_argument(
        "--static-cache-dir-name", default="static_dinov2_cls_patch_mean_cache"
    )
    parser.add_argument(
        "--static-cache-prefix", default="static_dinov2_cls_patch_mean"
    )
    parser.add_argument("--rag-corpus-path", default="rag_artifacts/action_rag_corpus.pt")
    parser.add_argument("--candidate-dir", default="rag_artifacts/candidates")
    parser.add_argument("--dynamic-checkpoint", default="/home/v100/ZYQ/skating/rag_results/dynamic/dynamic_seed2026_20260724_004400_best_epoch080_loss83.0444_spear0.8658.pth")
    parser.add_argument("--semantic-checkpoint", default="/home/v100/ZYQ/skating/rag_results/semantic/semantic_seed2026_20260723_160005_best_epoch050_mrr0.2570_goemae1.3456.pth")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--output-dir", default="rag_results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=5e-6)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rag-feature-dim", type=int, default=256)
    parser.add_argument("--baseline-static-proj-dim", type=int, default=512)
    parser.add_argument(
        "--baseline-head-type",
        choices=["metric", "legacy-womean"],
        default="metric",
    )
    parser.add_argument("--rag-delta-max", type=float, default=20.0)
    parser.add_argument("--delta-l2-weight", type=float, default=1e-4)
    parser.add_argument("--only-data-index", default=None)
    parser.add_argument("--only-train-data-index", default=None)
    parser.add_argument("--only-val-data-index", default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_state_dict(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint
    return checkpoint, {}


def load_checkpoint(
    model, path, device, strict, allowed_missing_prefixes=()
):
    state_dict, metadata = checkpoint_state_dict(path, device)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
    if strict:
        model.load_state_dict(state_dict, strict=True)
        print("loaded checkpoint strictly:", path)
    else:
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in current and current[key].shape == value.shape
        }
        result = model.load_state_dict(compatible, strict=False)
        ignored = sorted(set(state_dict).difference(compatible))
        invalid_missing = [
            key
            for key in result.missing_keys
            if not key.startswith(tuple(allowed_missing_prefixes))
        ]
        print("loaded compatible tensors:", len(compatible), "from", path)
        print("missing keys:", result.missing_keys)
        print("ignored checkpoint keys:", ignored)
        if ignored or invalid_missing:
            raise RuntimeError(
                "checkpoint is not baseline-compatible; ignored={} invalid_missing={}".format(
                    ignored, invalid_missing
                )
            )
    return metadata


def configure_trainable(model, stage):
    if stage == "dynamic":
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.rag.parameters():
            parameter.requires_grad = True
        for parameter in model.rag.semantic.parameters():
            parameter.requires_grad = False
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    print("trainable parameter groups:")
    for name, count in trainable:
        print(" ", name, count)
    print("trainable parameters:", sum(count for _, count in trainable))


def build_model(args, device):
    corpus = None
    use_rag = args.training_stage == "rag"
    if use_rag:
        corpus = load_action_corpus(args.rag_corpus_path)
        if not args.semantic_checkpoint:
            raise ValueError("--semantic-checkpoint is required for the rag stage")
        semantic_checkpoint = torch.load(args.semantic_checkpoint, map_location=device)
        semantic_config = semantic_checkpoint.get("config", {})
        semantic_model = FineFSSemanticRAG(
            coarse_classes=len(corpus["coarse_class_vocab"]),
            elements=len(corpus["element_vocab"]),
            query_dim=corpus["keys"].shape[1],
            evidence_dim=int(semantic_config.get("evidence_dim", 256)),
            encoder_hidden_dim=int(semantic_config.get("encoder_hidden_dim", 512)),
            metadata_dim=int(semantic_config.get("metadata_dim", 64)),
            temperature=float(semantic_config.get("temperature", 0.07)),
            dropout=float(semantic_config.get("dropout", 0.1)),
        ).to(device)
        semantic_model.load_state_dict(semantic_checkpoint["state_dict"], strict=True)
        semantic_model.eval()
        print("loaded frozen semantic checkpoint:", args.semantic_checkpoint)
        args.semantic_evidence_dim = int(semantic_config.get("evidence_dim", 256))
        args.semantic_encoder_hidden_dim = int(semantic_config.get("encoder_hidden_dim", 512))
        args.semantic_metadata_dim = int(semantic_config.get("metadata_dim", 64))
        args.semantic_temperature = float(semantic_config.get("temperature", 0.07))
        args.semantic_dropout = float(semantic_config.get("dropout", 0.1))
    else:
        semantic_model = None
    model = scoring_head(
        depth=2,
        input_dim=768,
        dim=512,
        input_len=16,
        num_scores=1,
        use_static_branch=use_rag,
        use_static_baseline=True,
        static_in_dim=2048,
        static_proj_dim=args.rag_feature_dim,
        baseline_static_proj_dim=args.baseline_static_proj_dim,
        baseline_head_type=args.baseline_head_type,
        time_score_dropout=0.2,
        rag_corpus=corpus,
        rag_semantic_model=semantic_model,
        rag_delta_max=args.rag_delta_max,
    ).to(device)
    if use_rag:
        if not args.dynamic_checkpoint:
            raise ValueError("--dynamic-checkpoint is required for the rag stage")
        metadata = load_checkpoint(
            model,
            args.dynamic_checkpoint,
            device=device,
            strict=False,
            allowed_missing_prefixes=("rag.",),
        )
        if metadata.get("training_stage") not in (None, "dynamic"):
            raise ValueError("RAG must start from a dynamic-stage checkpoint")
    elif args.init_checkpoint:
        load_checkpoint(model, args.init_checkpoint, device=device, strict=False)
    configure_trainable(model, args.training_stage)
    return model, corpus


def build_loaders(args, use_rag, device):
    common = dict(
        root_path=args.root_path,
        cache_dir_name=args.static_cache_dir_name,
        cache_prefix=args.static_cache_prefix,
        candidate_dir=args.candidate_dir if use_rag else None,
        require_candidates=use_rag,
    )
    train_filter = args.only_train_data_index or args.only_data_index
    val_filter = args.only_val_data_index or args.only_data_index
    train_dataset = FeatureDatasetWithStaticCache(
        is_train=True, only_data_index=train_filter, **common
    )
    val_dataset = FeatureDatasetWithStaticCache(
        is_train=False, only_data_index=val_filter, **common
    )
    if not train_dataset or not val_dataset:
        raise ValueError("train/validation dataset is empty")
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=av_collate_fn_with_static,
        pin_memory=device.type == "cuda",
        generator=train_generator,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=av_collate_fn_with_static,
        pin_memory=device.type == "cuda",
    )
    return train_loader, val_loader


def move_batch(batch, device):
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
    result = {
        "audio_feature": audio.to(device, non_blocking=True),
        "video_feature": video.to(device, non_blocking=True),
        "inv_audio_feature": inv_audio.to(device, non_blocking=True),
        "inv_video_feature": inv_video.to(device, non_blocking=True),
        "audio_len": audio_len,
        "video_len": video_len,
        "static_feature": static.to(device, non_blocking=True),
        "static_valid_mask": static_valid_mask.to(device, non_blocking=True),
        "target": scores[0].to(device, non_blocking=True),
        "data_indices": data_indices,
    }
    if rag_data is not None:
        result["candidate_indices"] = rag_data["candidate_indices"].to(
            device, non_blocking=True
        )
        result["candidate_similarities"] = rag_data[
            "candidate_similarities"
        ].to(device, non_blocking=True)
        result["overlap_weights"] = rag_data["overlap_weights"].to(
            device, non_blocking=True
        )
    else:
        result["candidate_indices"] = None
        result["candidate_similarities"] = None
        result["overlap_weights"] = None
    return result


def forward_batch(model, batch):
    return model(
        batch["audio_feature"],
        batch["video_feature"],
        batch["inv_audio_feature"],
        batch["inv_video_feature"],
        batch["audio_len"],
        batch["video_len"],
        static_feature=batch["static_feature"],
        static_valid_mask=batch["static_valid_mask"],
        candidate_indices=batch["candidate_indices"],
        candidate_similarities=batch["candidate_similarities"],
        overlap_weights=batch["overlap_weights"],
    )


def metrics(truth, prediction):
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    return {
        "mse": float(np.mean((prediction - truth) ** 2)),
        "mae": float(np.mean(np.abs(prediction - truth))),
        "spearman": float(spearmanr(truth, prediction).correlation),
    }


def run_epoch(loader, model, device, optimizer=None, delta_l2_weight=0.0, limit=None):
    training = optimizer is not None
    if not training:
        model.eval()
    elif model.rag is not None:
        model.eval()
        model.rag.train()
    else:
        model.train()
    truth = []
    final_prediction = []
    baseline_prediction = []
    deltas = []
    valid_evidence = []
    total_loss = 0.0
    samples = 0
    for batch_index, raw_batch in enumerate(loader):
        if limit is not None and batch_index >= limit:
            break
        batch = move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = forward_batch(model, batch)
            final_loss = torch.nn.functional.mse_loss(output["score"], batch["target"])
            delta_regularizer = output["delta_tes_rag"].pow(2).mean()
            loss = final_loss + delta_l2_weight * delta_regularizer
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 5.0
                )
                optimizer.step()
        batch_size = batch["target"].shape[0]
        total_loss += float(loss.detach()) * batch_size
        samples += batch_size
        truth.extend(batch["target"].detach().cpu().tolist())
        final_prediction.extend(output["score"].detach().cpu().tolist())
        baseline_prediction.extend(output["tes_baseline"].detach().cpu().tolist())
        deltas.extend(output["delta_tes_rag"].detach().cpu().tolist())
        if "evidence_valid_mask" in output:
            valid_evidence.extend(
                output["evidence_valid_mask"].detach().cpu().float().tolist()
            )
    if samples == 0:
        raise ValueError("epoch processed zero samples")
    result = {
        "loss": total_loss / samples,
        "final": metrics(truth, final_prediction),
        "baseline": metrics(truth, baseline_prediction),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
        "evidence_valid_ratio": float(np.mean(valid_evidence)) if valid_evidence else 0.0,
        "samples": samples,
    }
    return result


def save_checkpoint(path, model, args, corpus, epoch, validation):
    state = {
        "state_dict": model.state_dict(),
        "training_stage": args.training_stage,
        "epoch": epoch,
        "validation": validation,
        "config": vars(args),
        "corpus_version": None if corpus is None else corpus["metadata_version"],
    }
    torch.save(state, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = args.run_name or f"{args.training_stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = os.path.join(args.output_dir, run_name + ".jsonl")
    model, corpus = build_model(args, device)
    train_loader, val_loader = build_loaders(
        args, args.training_stage == "rag", device
    )
    learning_rate = args.lr
    if learning_rate is None:
        learning_rate = 1e-4 if args.training_stage == "dynamic" else 3e-4
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    best_spearman = -float("inf")
    best_path = None
    print("device:", device)
    print("train samples:", len(train_loader.dataset))
    print("validation samples:", len(val_loader.dataset))
    print("log:", log_path)

    with open(log_path, "w", encoding="utf-8") as log_handle:
        initial_validation = run_epoch(
            val_loader,
            model,
            device,
            optimizer=None,
            limit=args.limit_val_batches,
        )
        initial_record = {"epoch": -1, "initial_validation": initial_validation}
        print(json.dumps(initial_record, ensure_ascii=False))
        log_handle.write(json.dumps(initial_record, ensure_ascii=False) + "\n")
        log_handle.flush()
        for epoch in range(args.epochs):
            train_result = run_epoch(
                train_loader,
                model,
                device,
                optimizer=optimizer,
                delta_l2_weight=(
                    args.delta_l2_weight if args.training_stage == "rag" else 0.0
                ),
                limit=args.limit_train_batches,
            )
            val_result = run_epoch(
                val_loader,
                model,
                device,
                optimizer=None,
                limit=args.limit_val_batches,
            )
            record = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": train_result,
                "validation": val_result,
            }
            print(json.dumps(record, ensure_ascii=False))
            log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_handle.flush()
            current_spearman = val_result["final"]["spearman"]
            is_better = best_path is None or (
                np.isfinite(current_spearman)
                and (
                    not np.isfinite(best_spearman)
                    or current_spearman > best_spearman
                )
            )
            if is_better:
                best_spearman = current_spearman
                best_path = os.path.join(
                    args.output_dir,
                    (
                        f"{run_name}_best_epoch{epoch + 1:03d}"
                        f"_loss{val_result['loss']:.4f}"
                        f"_spear{current_spearman:.4f}.pth"
                    ),
                )
                save_checkpoint(
                    best_path, model, args, corpus, epoch, val_result
                )
                print("saved best checkpoint:", best_path)
            scheduler.step()
    print("best validation spearman:", best_spearman)
    print("best checkpoint:", best_path)


if __name__ == "__main__":
    main()
