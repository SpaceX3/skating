from typing_extensions import final
import argparse
import json
import random
import torch
import torch.utils.data as data
import os
import sys
import numpy as np
from model import scoring_head, head
from dataset.dataset_fs800 import FeatureDatasetWithStaticCache, av_collate_fn_with_static
from scipy.stats import spearmanr 
import math
from datetime import datetime
# from torch.optim import lr_sheduler
# import time
# import warnings


def set_trainable_params_for_stage(model, stage: int, freeze_static_proj_in_stage2: bool = True):
    """
    stage=1: freeze dynamic backbone, only train static_proj + time_score_mlp
    stage=2: train dynamic backbone + time_score_mlp, optional static_proj freeze
    """
    if stage == 1:
        for name, p in model.named_parameters():
            p.requires_grad = any(
                component in name
                for component in ("query_memory_fusion", "static_proj", "time_score_mlp")
            )
    else:
        for name, p in model.named_parameters():
            if freeze_static_proj_in_stage2 and ("static_proj" in name):
                p.requires_grad = False
            else:
                p.requires_grad = True


def build_optimizer_and_scheduler(model, lr=1e-4, weight_decay=5e-6, step_size=20, gamma=0.7):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return optimizer, scheduler


dev = 0


def checkpoint_path(output_dir, kind):
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, "best_{}.pth".format(kind))


def save_best_checkpoint(model, save_path):
    torch.save(model.state_dict(), save_path)
    return save_path


def build_model(static_in_dim=1536):
    return scoring_head(
        depth=2,
        input_dim=768,
        dim=512,
        input_len=16,
        num_scores=1,
        use_static_branch=True,
        static_in_dim=static_in_dim,
        static_proj_dim=128,
        use_query_memory_fusion=True,
    )


def load_dynamic_checkpoint(model, checkpoint_path):
    excluded_prefixes = (
        "query_memory_fusion.",
        "static_proj.",
        "time_score_mlp.",
    )
    checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
    model_state = model.state_dict()
    expected_keys = {
        key for key in model_state if not key.startswith(excluded_prefixes)
    }
    dynamic_state = {
        key: value
        for key, value in checkpoint_state.items()
        if not key.startswith(excluded_prefixes)
    }

    missing_keys = sorted(expected_keys - dynamic_state.keys())
    unexpected_keys = sorted(dynamic_state.keys() - expected_keys)
    shape_mismatches = sorted(
        key
        for key in expected_keys & dynamic_state.keys()
        if dynamic_state[key].shape != model_state[key].shape
    )
    if missing_keys or unexpected_keys or shape_mismatches:
        raise ValueError(
            "dynamic checkpoint mismatch: missing={}, unexpected={}, shape={}".format(
                missing_keys, unexpected_keys, shape_mismatches
            )
        )

    model.load_state_dict(dynamic_state, strict=False)
    return sorted(dynamic_state)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_log_output(log_dir, log_file=None):
    os.makedirs(log_dir, exist_ok=True)
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    else:
        log_file = os.path.abspath(log_file)
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    log_stream = open(log_file, "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_stream)
    sys.stderr = Tee(sys.__stderr__, log_stream)
    print(f"log file: {log_file}")
    return log_stream

def validation(dataloader, model, criterion, score_index):
    model.eval()
    val_loss = 0
    val_truth = []
    val_pred = []


    for audio_feature, video_feature, inv_audio_feature, inv_video_feature, static_feature, audio_len, video_len, score, data_index in dataloader:
        batch_size, _, _, _ = audio_feature.shape
        audio_feature = audio_feature.cuda(device=dev)
        video_feature = video_feature.cuda(device=dev)
        inv_audio_feature = inv_audio_feature.cuda(device=dev)
        inv_video_feature = inv_video_feature.cuda(device=dev)
        static_feature = static_feature.cuda(device=dev)
        target = score[score_index].cuda(device=dev)

        with torch.no_grad():
            output = model(audio_feature, video_feature, inv_audio_feature, inv_video_feature, audio_len, video_len, static_feature)
        val_pred.append(output.detach().data.cpu().numpy())
        val_truth.append(target.cpu().numpy())

        loss = criterion(output, target)

        val_loss += loss.item() * batch_size

    val_truth = np.concatenate(val_truth)
    val_pred = np.concatenate(val_pred)
    spear = spearmanr(val_truth, val_pred)
    print(len(dataloader.dataset))
    val_loss = val_loss / len(dataloader.dataset)
    return val_loss, spear.correlation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--root-path", default="/home/v100/ZYQ/FS1000 Dataset")
    parser.add_argument(
        "--static-cache-dir",
        default="/media/v100/disk3t/skating/fs1000_static_videomae_c1_class_retrieval",
    )
    parser.add_argument("--static-cache-prefix", default="static_videomae_c1_class_retrieval")
    parser.add_argument("--static-feature-dim", type=int, default=1536)
    parser.add_argument(
        "--log-dir",
        default="/media/v100/disk3t/skating/experiments/videomae_c1_class_retrieval/manual_seed2026",
    )
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--init-dynamic-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    dev = args.gpu
    seed_everything(args.seed)
    log_stream = setup_log_output(args.log_dir, args.log_file)
    with open(os.path.join(args.log_dir, "resolved_config.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
        handle.write("\n")

    # build dataset
    train_dataset = FeatureDatasetWithStaticCache(
        root_path=args.root_path,
        is_train=True,
        cache_dir_name=args.static_cache_dir,
        cache_prefix=args.static_cache_prefix,
        static_feature_dim=args.static_feature_dim,
    )
    val_dataset = FeatureDatasetWithStaticCache(
        root_path=args.root_path,
        is_train=False,
        cache_dir_name=args.static_cache_dir,
        cache_prefix=args.static_cache_prefix,
        static_feature_dim=args.static_feature_dim,
    )

    generator = torch.Generator().manual_seed(args.seed)
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=av_collate_fn_with_static, generator=generator)
    val_dataloader = data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=av_collate_fn_with_static)

    # model
    model = build_model(args.static_feature_dim)
    if args.init_dynamic_checkpoint:
        loaded_keys = load_dynamic_checkpoint(model, args.init_dynamic_checkpoint)
        print(
            "loaded {} dynamic parameters from {}; query_memory_fusion, "
            "static_proj, and time_score_mlp remain newly initialized".format(
                len(loaded_keys), args.init_dynamic_checkpoint
            )
        )
    model = model.cuda(device=dev)

    epochs = args.epochs
    warm_up_epochs = 10

    # Two-stage training: first fit the new static branch and score head, then fine-tune all parameters.
    freeze_backbone_epochs = args.freeze_backbone_epochs
    freeze_static_proj_in_stage2 = False
    set_trainable_params_for_stage(
        model,
        stage=1 if freeze_backbone_epochs > 0 else 2,
        freeze_static_proj_in_stage2=freeze_static_proj_in_stage2
    )
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr=1e-4, weight_decay=5e-6, step_size=30, gamma=0.9
    )
    if freeze_backbone_epochs > 0:
        print(
            f"Stage-1 enabled: train query_memory_fusion + static_proj + "
            f"time_score_mlp for first "
            f"{freeze_backbone_epochs} epochs."
        )

    # criterion
    criterion = torch.nn.MSELoss()

    # other parameter
    score_index = 0

    min_val_loss = 10000
    max_spear_cor = -float("inf")
    best_loss_checkpoint = None
    best_spear_checkpoint = None

    for epoch_idx in range(epochs):
        if freeze_backbone_epochs > 0 and epoch_idx == freeze_backbone_epochs:
            set_trainable_params_for_stage(model, stage=2, freeze_static_proj_in_stage2=freeze_static_proj_in_stage2)
            optimizer, scheduler = build_optimizer_and_scheduler(
                model, lr=1e-4, weight_decay=5e-6, step_size=20, gamma=0.7
            )
            print("Stage-2 enabled: train all parameters, including static_proj.")

        model.train()
        print("="*25)
        print("epoch ", epoch_idx)
        epoch_train_loss = 0.0
        train_truth = []
        train_pred = []
        
        for audio_feature, video_feature, inv_audio_feature, inv_video_feature, static_feature, audio_len, video_len, score, data_index in train_dataloader:
            batch_size, _, _, _ = audio_feature.shape
            audio_feature = audio_feature.cuda(device=dev)
            video_feature = video_feature.cuda(device=dev)
            inv_audio_feature = inv_audio_feature.cuda(device=dev)
            inv_video_feature = inv_video_feature.cuda(device=dev)
            static_feature = static_feature.cuda(device=dev)
            
            target = score[score_index].cuda(device=dev)

            train_loss = 0
            optimizer.zero_grad()

            output = model(audio_feature, video_feature, inv_audio_feature, inv_video_feature, audio_len, video_len, static_feature)

            loss = criterion(output, target)
            train_loss = loss.item()
            epoch_train_loss += train_loss * batch_size
            train_pred.append(output.detach().cpu().numpy())
            train_truth.append(target.detach().cpu().numpy())

            loss.backward()
            optimizer.step()
        train_truth = np.concatenate(train_truth)
        train_pred = np.concatenate(train_pred)
        train_spear = spearmanr(train_truth, train_pred).correlation
        epoch_train_loss = epoch_train_loss / len(train_dataloader.dataset)
        print("train_loss: ", epoch_train_loss, " | train spear corr: ", train_spear)
        scheduler.step()
        # validation
        val_loss, spear = validation(val_dataloader, model, criterion, score_index)
        print("val_loss: ", val_loss, " | spear corr: ", spear)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_path = checkpoint_path(args.log_dir, "loss")
            best_loss_checkpoint = save_best_checkpoint(model, save_path)
            print("saved best loss checkpoint: ", save_path)
        if spear > max_spear_cor:
            max_spear_cor = spear
            save_path = checkpoint_path(args.log_dir, "spearman")
            best_spear_checkpoint = save_best_checkpoint(model, save_path)
            print("saved best spear checkpoint: ", save_path)
            
        print("min validation loss: ", min_val_loss, " | max spear corr: ", max_spear_cor)
        print("checkpoint")
        
        
        print(optimizer.param_groups[0]['lr'])
        torch.save(model.state_dict(), os.path.join(args.log_dir, "last.pth"))
        with open(os.path.join(args.log_dir, "best_metrics.json"), "w") as handle:
            json.dump(
                {
                    "best_val_loss": min_val_loss,
                    "best_val_spearman": max_spear_cor,
                    "last_epoch": epoch_idx,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        print()
        # scheduler.step()
        
        
