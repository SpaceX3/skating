from typing_extensions import final
import argparse
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
            p.requires_grad = ("static_proj" in name) or ("time_score_mlp" in name)
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


def checkpoint_path(epoch_idx, val_loss, spear):
    output_dir = "./fs800_result"
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, "checkpoint_epoch{}_loss{:.2f}_spear{:.3f}.pth".format(
        epoch_idx,
        val_loss,
        spear,
    ))


def save_best_checkpoint(model, save_path, previous_path):
    torch.save(model.state_dict(), save_path)
    if previous_path and previous_path != save_path and os.path.exists(previous_path):
        os.remove(previous_path)
    return save_path


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


def close_log_output(log_stream):
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_stream.close()

def validation(dataloader, model, criterion, score_index, gpu):
    model.eval()
    val_loss = 0
    val_truth = []
    val_pred = []


    for audio_feature, video_feature, inv_audio_feature, inv_video_feature, static_feature, audio_len, video_len, score, data_index in dataloader:
        batch_size, _, _, _ = audio_feature.shape
        audio_feature = audio_feature.cuda(device=gpu)
        video_feature = video_feature.cuda(device=gpu)
        inv_audio_feature = inv_audio_feature.cuda(device=gpu)
        inv_video_feature = inv_video_feature.cuda(device=gpu)
        static_feature = static_feature.cuda(device=gpu)
        target = score[score_index].cuda(device=gpu)

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
    parser.add_argument("--log-dir", default="./fs800_result")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--root-path", default="../FS1000 Dataset/")
    parser.add_argument("--static-cache-dir-name", default="static_resnet50_keyframe_cache") # static_resnet50_cache
    parser.add_argument("--static-cache-prefix", default="static_resnet50_keyframe")  # static_resnet50
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=30)
    parser.add_argument("--freeze-static-proj-in-stage2", action="store_true")
    parser.add_argument("--score-index", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-6)
    parser.add_argument("--only-data-index", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    gpu = args.gpu
    log_stream = setup_log_output(args.log_dir, args.log_file)

    # build dataset
    train_dataset = FeatureDatasetWithStaticCache(
        root_path=args.root_path,
        is_train=True,
        cache_dir_name=args.static_cache_dir_name,
        cache_prefix=args.static_cache_prefix,
        only_data_index=args.only_data_index,
    )
    val_dataset = FeatureDatasetWithStaticCache(
        root_path=args.root_path,
        is_train=False,
        cache_dir_name=args.static_cache_dir_name,
        cache_prefix=args.static_cache_prefix,
        only_data_index=args.only_data_index,
    )

    loader_workers = 0 if args.smoke_test else args.num_workers
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=loader_workers, shuffle=(len(train_dataset) > 0), collate_fn=av_collate_fn_with_static)
    val_dataloader = data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=loader_workers, collate_fn=av_collate_fn_with_static)

    if args.smoke_test:
        smoke_loader = train_dataloader if len(train_dataset) > 0 else val_dataloader
        if len(smoke_loader.dataset) == 0:
            raise ValueError("Smoke test dataset is empty. Check --only-data-index and split files.")
        batch = next(iter(smoke_loader))
        static_feature = batch[4]
        audio_feature = batch[0]
        video_feature = batch[1]
        print("smoke data_index:", batch[-1])
        print("smoke audio shape:", tuple(audio_feature.shape))
        print("smoke video shape:", tuple(video_feature.shape))
        print("smoke static shape:", tuple(static_feature.shape))
        print("smoke static cache dir:", args.static_cache_dir_name)
        print("smoke static cache prefix:", args.static_cache_prefix)
        close_log_output(log_stream)
        sys.exit(0)

    # model
    model = scoring_head(
        depth=2,
        input_dim=768,
        dim=512,
        input_len=16,
        num_scores=1,
        use_static_branch=True,
        static_in_dim=2048,
        static_proj_dim=128,
    ).cuda(device=gpu)  #, bidirection=True

    epochs = args.epochs
    warm_up_epochs = 10

    # two-stage training: stage-1 only trains time_score_mlp, stage-2 trains all params
    freeze_backbone_epochs = args.freeze_backbone_epochs
    freeze_static_proj_in_stage2 = args.freeze_static_proj_in_stage2
    set_trainable_params_for_stage(
        model,
        stage=1 if freeze_backbone_epochs > 0 else 2,
        freeze_static_proj_in_stage2=freeze_static_proj_in_stage2
    )
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr=args.lr, weight_decay=args.weight_decay, step_size=30, gamma=0.9
    )
    if freeze_backbone_epochs > 0:
        print(f"Stage-1 enabled: train only time_score_mlp for first {freeze_backbone_epochs} epochs.")

    # criterion
    criterion = torch.nn.MSELoss()

    # other parameter
    score_index = args.score_index

    min_val_loss = 10000
    max_spear_cor = 0
    best_loss_checkpoint = None
    best_spear_checkpoint = None

    for epoch_idx in range(epochs):
        if freeze_backbone_epochs > 0 and epoch_idx == freeze_backbone_epochs:
            set_trainable_params_for_stage(model, stage=2, freeze_static_proj_in_stage2=freeze_static_proj_in_stage2)
            optimizer, scheduler = build_optimizer_and_scheduler(
                model, lr=args.lr, weight_decay=args.weight_decay, step_size=20, gamma=0.7
            )
            static_msg = "keep static_proj frozen" if freeze_static_proj_in_stage2 else "train static_proj"
            print(f"Stage-2 enabled: train dynamic backbone + time head, {static_msg}.")

        model.train()
        print("="*25)
        print("epoch ", epoch_idx)
        epoch_train_loss = 0.0
        train_truth = []
        train_pred = []

        for audio_feature, video_feature, inv_audio_feature, inv_video_feature, static_feature, audio_len, video_len, score, data_index in train_dataloader:
            batch_size, _, _, _ = audio_feature.shape
            audio_feature = audio_feature.cuda(device=gpu)
            video_feature = video_feature.cuda(device=gpu)
            inv_audio_feature = inv_audio_feature.cuda(device=gpu)
            inv_video_feature = inv_video_feature.cuda(device=gpu)
            static_feature = static_feature.cuda(device=gpu)

            target = score[score_index].cuda(device=gpu)

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
        val_loss, spear = validation(val_dataloader, model, criterion, score_index, gpu)
        print("val_loss: ", val_loss, " | spear corr: ", spear)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_path = checkpoint_path(epoch_idx, val_loss, spear)
            best_loss_checkpoint = save_best_checkpoint(model, save_path, best_loss_checkpoint)
            print("saved best loss checkpoint: ", save_path)
        if spear > max_spear_cor:
            max_spear_cor = spear
            save_path = checkpoint_path(epoch_idx, val_loss, spear)
            best_spear_checkpoint = save_best_checkpoint(model, save_path, best_spear_checkpoint)
            print("saved best spear checkpoint: ", save_path)

        print("min validation loss: ", min_val_loss, " | max spear corr: ", max_spear_cor)
        print("checkpoint")


        print(optimizer.param_groups[0]['lr'])
        # scheduler.step()

