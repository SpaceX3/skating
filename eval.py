import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data as data
from scipy.stats import spearmanr

from dataset.dataset_fs800 import FeatureDatasetWithStaticCache, av_collate_fn_with_static
from model import scoring_head

dev=0


def load_state_dict(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if any(key.startswith("module.") for key in state_dict):
        stripped = OrderedDict()
        for key, value in state_dict.items():
            stripped[key.removeprefix("module.")] = value
        state_dict = stripped

    return state_dict


def validate(dataloader, model, criterion, score_index, device):
    model.eval()
    val_loss = 0.0
    val_truth = []
    val_pred = []

    with torch.no_grad():
        for (
            audio_feature,
            video_feature,
            inv_audio_feature,
            inv_video_feature,
            static_feature,
            audio_len,
            video_len,
            score,
            data_index,
        ) in dataloader:
            batch_size = audio_feature.shape[0]
            audio_feature = audio_feature.to(device)
            video_feature = video_feature.to(device)
            inv_audio_feature = inv_audio_feature.to(device)
            inv_video_feature = inv_video_feature.to(device)
            static_feature = static_feature.to(device)
            target = score[score_index].to(device)

            output = model(
                audio_feature,
                video_feature,
                inv_audio_feature,
                inv_video_feature,
                audio_len,
                video_len,
                static_feature,
            )
            loss = criterion(output, target)

            val_loss += loss.item() * batch_size
            #val_pred.append(output.detach().cpu().numpy())
            val_pred.append(output.detach().data.cpu().numpy())
            val_truth.append(target.detach().cpu().numpy())

    val_truth = np.concatenate(val_truth)
    val_pred = np.concatenate(val_pred)
    correlation = spearmanr(val_truth, val_pred).correlation
    return val_loss / len(dataloader.dataset), correlation


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/home/v100/ZYQ/skating/fs800_result/checkpoint_epoch102_loss61.58_spear0.859.pth",
    )
    parser.add_argument("--root-path", default="../FS1000 Dataset/")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--score-index", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    val_dataset = FeatureDatasetWithStaticCache(root_path=args.root_path, is_train=False)
    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=av_collate_fn_with_static,
        pin_memory=device.type == "cuda",
    )

    model = scoring_head(
        depth=2,
        input_dim=768,
        dim=512,
        input_len=16,
        num_scores=1,
        use_static_branch=True,
        static_in_dim=2048,
        static_proj_dim=128,
    ).to(device)

    state_dict = load_state_dict(args.checkpoint, device)
    model.load_state_dict(state_dict, strict=True)

    criterion = torch.nn.MSELoss()
    # val_loss, spear = validate(val_dataloader, model, criterion, args.score_index, device)
    val_loss, spear = validation(val_dataloader, model, criterion, args.score_index)
    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}")
    print(f"validation samples: {len(val_dataset)}")
    print(f"val_loss: {val_loss}")
    print(f"spear corr: {spear}")


if __name__ == "__main__":
    main()
