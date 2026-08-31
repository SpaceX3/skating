#!/usr/bin/env python3
"""Train and evaluate the Top-4 Cross-Attention model on FineFS TES."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.stats
import torch
import torch.utils.data as data

from dataset.dataset_finefs_top4 import (
    FineFSFeatureDatasetWithStaticCache,
    av_collate_fn_with_static,
)
from model import scoring_head


DEFAULT_SPLIT_JSON = (
    "/media/v100/disk3t/finefs_pocr_classifier/experiments/"
    "e11_video_grain_split/video_split.json"
)
DEFAULT_ANNOTATION_DIR = "/home/v100/ZYQ/FineFS/annotation"
DEFAULT_AUDIO_FEATURE_DIR = (
    "/home/v100/ZYQ/finefs_av_feature_extractor/features/ast_feature_finefs"
)
DEFAULT_VIDEO_FEATURE_DIR = (
    "/home/v100/ZYQ/finefs_av_feature_extractor/features/"
    "Timesformer_output_feature_finefs"
)
DEFAULT_STATIC_CACHE_DIR = (
    "/media/v100/disk3t/skating/"
    "finefs_static_videomae_c1_top4_cross_attention"
)
DEFAULT_STATIC_CACHE_PREFIX = "static_videomae_c1_top4_cross_attention"
DEFAULT_STATIC_FEATURE_DIM = 6914


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_log_output(log_dir: Path, log_file: Optional[str]):
    log_dir.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        log_file = str(log_dir / ("train_{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S"))))
    log_path = Path(log_file).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, stream)
    sys.stderr = Tee(sys.__stderr__, stream)
    print("log file: {}".format(log_path))
    return stream


def build_model(static_feature_dim: int):
    return scoring_head(
        depth=2,
        input_dim=768,
        dim=512,
        input_len=16,
        num_scores=1,
        use_static_branch=True,
        static_in_dim=int(static_feature_dim),
        static_proj_dim=128,
        use_top4_cross_attention=True,
    )


def _state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a state dict")
    if any(key.startswith("module.") for key in checkpoint):
        return OrderedDict(
            (key[7:] if key.startswith("module.") else key, value)
            for key, value in checkpoint.items()
        )
    return checkpoint


def load_dynamic_checkpoint(model, checkpoint_path: str, device: torch.device):
    checkpoint = _state_dict(torch.load(checkpoint_path, map_location=device))
    excluded = (
        "query_support_cross_attention.",
        "static_proj.",
        "time_score_mlp.",
    )
    model_state = model.state_dict()
    expected = {key for key in model_state if not key.startswith(excluded)}
    dynamic = {key: value for key, value in checkpoint.items() if not key.startswith(excluded)}
    missing = sorted(expected - dynamic.keys())
    unexpected = sorted(dynamic.keys() - expected)
    mismatched = sorted(
        key for key in expected & dynamic.keys() if dynamic[key].shape != model_state[key].shape
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "dynamic checkpoint mismatch: missing={}, unexpected={}, shape={}".format(
                missing, unexpected, mismatched
            )
        )
    model.load_state_dict(dynamic, strict=False)
    return len(dynamic)


def set_trainable_params_for_stage(model, stage: int):
    if stage == 1:
        trainable_components = (
            "query_support_cross_attention",
            "static_proj",
            "time_score_mlp",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(
                component in name for component in trainable_components
            )
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True


def build_optimizer_and_scheduler(model, learning_rate, weight_decay, step_size, gamma):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("no trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return optimizer, scheduler


def make_dataset(args, split):
    return FineFSFeatureDatasetWithStaticCache(
        split_json=args.split_json,
        annotation_dir=args.annotation_dir,
        audio_feature_dir=args.audio_feature_dir,
        video_feature_dir=args.video_feature_dir,
        static_cache_dir=args.static_cache_dir,
        static_cache_prefix=args.static_cache_prefix,
        static_feature_dim=args.static_feature_dim,
        split=split,
    )


def make_loader(dataset, args, shuffle=False):
    return data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=av_collate_fn_with_static,
        pin_memory=args.device.type == "cuda",
    )


def _move_batch(batch, device):
    audio, video, inv_audio, inv_video, static, audio_len, video_len, scores, ids = batch
    return (
        audio.to(device, non_blocking=True),
        video.to(device, non_blocking=True),
        inv_audio.to(device, non_blocking=True),
        inv_video.to(device, non_blocking=True),
        static.to(device, non_blocking=True),
        audio_len,
        video_len,
        scores[0].to(device, non_blocking=True),
        ids,
    )


def regression_metrics(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return {
        "mse": float(np.mean((truth - prediction) ** 2)),
        "spearman": float(scipy.stats.spearmanr(truth, prediction).correlation),
        "pearson": float(scipy.stats.pearsonr(truth, prediction)[0]),
    }


def train_one_epoch(loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    truth = []
    prediction = []
    for batch in loader:
        audio, video, inv_audio, inv_video, static, audio_len, video_len, target, _ = _move_batch(
            batch, device
        )
        optimizer.zero_grad()
        output = model(audio, video, inv_audio, inv_video, audio_len, video_len, static)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * target.shape[0]
        truth.append(target.detach().cpu().numpy())
        prediction.append(output.detach().cpu().numpy())
    metrics = regression_metrics(np.concatenate(truth), np.concatenate(prediction))
    metrics["mse"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    truth = []
    prediction = []
    for batch in loader:
        audio, video, inv_audio, inv_video, static, audio_len, video_len, target, _ = _move_batch(
            batch, device
        )
        output = model(audio, video, inv_audio, inv_video, audio_len, video_len, static)
        total_loss += criterion(output, target).item() * target.shape[0]
        truth.append(target.cpu().numpy())
        prediction.append(output.cpu().numpy())
    metrics = regression_metrics(np.concatenate(truth), np.concatenate(prediction))
    metrics["mse"] = total_loss / len(loader.dataset)
    return metrics


def train(args):
    log_stream = setup_log_output(Path(args.log_dir), args.log_file)
    try:
        with (Path(args.log_dir) / "resolved_config.json").open("w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")

        train_dataset = make_dataset(args, "train")
        val_dataset = make_dataset(args, "val")
        train_loader = make_loader(train_dataset, args, shuffle=True)
        val_loader = make_loader(val_dataset, args)
        model = build_model(args.static_feature_dim)
        if args.init_dynamic_checkpoint:
            loaded = load_dynamic_checkpoint(model, args.init_dynamic_checkpoint, args.device)
            print(
                "loaded {} dynamic parameters from {}; static Cross-Attention, projection, "
                "and temporal classifier remain newly initialized".format(
                    loaded, args.init_dynamic_checkpoint
                )
            )
        model.to(args.device)
        criterion = torch.nn.MSELoss()
        set_trainable_params_for_stage(
            model, 1 if args.freeze_backbone_epochs > 0 else 2
        )
        optimizer, scheduler = build_optimizer_and_scheduler(
            model,
            args.learning_rate,
            args.weight_decay,
            args.scheduler_step,
            args.scheduler_gamma,
        )
        if args.freeze_backbone_epochs > 0:
            print(
                "Stage-1: train Cross-Attention + static projection + temporal MLP for "
                "{} epochs".format(args.freeze_backbone_epochs)
            )
        print("train videos: {}".format(len(train_dataset)))
        print("val videos: {}".format(len(val_dataset)))
        print("target: total_element_score (TES)")
        print("device: {}".format(args.device))
        print()

        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        best_mse = float("inf")
        best_spearman = -float("inf")
        for epoch in range(args.epochs):
            if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs:
                set_trainable_params_for_stage(model, 2)
                optimizer, scheduler = build_optimizer_and_scheduler(
                    model,
                    args.stage2_learning_rate,
                    args.weight_decay,
                    args.stage2_scheduler_step,
                    args.stage2_scheduler_gamma,
                )
                print("Stage-2: train all model parameters")

            train_metrics = train_one_epoch(
                train_loader, model, optimizer, criterion, args.device
            )
            val_metrics = evaluate(val_loader, model, criterion, args.device)
            scheduler.step()
            print("=" * 25)
            print("epoch {}".format(epoch))
            print(
                "train_loss: {:.6f} | train spear corr: {:.6f}".format(
                    train_metrics["mse"], train_metrics["spearman"]
                )
            )
            print(
                "val_loss: {:.6f} | val spear corr: {:.6f} | val pearson corr: {:.6f}".format(
                    val_metrics["mse"], val_metrics["spearman"], val_metrics["pearson"]
                )
            )
            if val_metrics["mse"] < best_mse:
                best_mse = val_metrics["mse"]
                torch.save(model.state_dict(), log_dir / "best_loss.pth")
                print("saved best_loss.pth")
            if val_metrics["spearman"] > best_spearman:
                best_spearman = val_metrics["spearman"]
                torch.save(model.state_dict(), log_dir / "best_spearman.pth")
                print("saved best_spearman.pth")
            torch.save(model.state_dict(), log_dir / "last.pth")
            with (log_dir / "best_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "best_val_mse": best_mse,
                        "best_val_spearman": best_spearman,
                        "last_epoch": epoch,
                    },
                    handle,
                    indent=2,
                )
                handle.write("\n")
            print("learning_rate: {}\n".format(optimizer.param_groups[0]["lr"]))
    finally:
        log_stream.close()


def test(args):
    test_dataset = make_dataset(args, "test")
    test_loader = make_loader(test_dataset, args)
    model = build_model(args.static_feature_dim).to(args.device)
    checkpoint = _state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.load_state_dict(checkpoint, strict=True)
    metrics = evaluate(test_loader, model, torch.nn.MSELoss(), args.device)
    print("test videos: {}".format(len(test_dataset)))
    print("target: total_element_score (TES)")
    print("device: {}".format(args.device))
    print("checkpoint: {}".format(args.checkpoint))
    print(
        "test MSE={:.6f} Spearman={:.6f} Pearson={:.6f}".format(
            metrics["mse"], metrics["spearman"], metrics["pearson"]
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="FineFS Top-4 Cross-Attention TES regression")
    parser.add_argument("--mode", choices=("train", "test"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split-json", default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--annotation-dir", default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--audio-feature-dir", default=DEFAULT_AUDIO_FEATURE_DIR)
    parser.add_argument("--video-feature-dir", default=DEFAULT_VIDEO_FEATURE_DIR)
    parser.add_argument("--static-cache-dir", default=DEFAULT_STATIC_CACHE_DIR)
    parser.add_argument("--static-cache-prefix", default=DEFAULT_STATIC_CACHE_PREFIX)
    parser.add_argument("--static-feature-dim", type=int, default=DEFAULT_STATIC_FEATURE_DIM)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--init-dynamic-checkpoint", default=None)
    parser.add_argument("--log-dir", default="/media/v100/disk3t/skating/experiments/finefs_top4_cross_attention/seed2026_warmstart")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--stage2-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-6)
    parser.add_argument("--scheduler-step", type=int, default=30)
    parser.add_argument("--scheduler-gamma", type=float, default=0.9)
    parser.add_argument("--stage2-scheduler-step", type=int, default=20)
    parser.add_argument("--stage2-scheduler-gamma", type=float, default=0.7)
    args = parser.parse_args()
    if args.mode == "test" and not args.checkpoint:
        parser.error("--checkpoint is required in test mode")
    args.device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.mode == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
