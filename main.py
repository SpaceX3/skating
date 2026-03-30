import argparse
import torch
import torch.utils.data as data
import numpy as np
from model import scoring_head
from dataset.dataset_fs800 import FeatureDatasetWithStaticCache, av_collate_fn_with_static
from scipy.stats import spearmanr 

dev = 0

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
    parser.add_argument("--gpu", type=int, default=dev)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--strict_static_cache", action="store_true")
    parser.add_argument("--allow_missing_static_cache", action="store_true")
    args = parser.parse_args()

    dev = args.gpu

    # build dataset
    train_dataset = FeatureDatasetWithStaticCache(
        root_path='../FS1000 Dataset/',
        is_train=True,
        strict_cache=(not args.allow_missing_static_cache),
    )
    val_dataset = FeatureDatasetWithStaticCache(
        root_path='../FS1000 Dataset/',
        is_train=False,
        strict_cache=(not args.allow_missing_static_cache),
    )

    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=16,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=av_collate_fn_with_static,
    )
    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=16,
        num_workers=args.num_workers,
        collate_fn=av_collate_fn_with_static,
    )

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
        fusion_dropout=0.3,
        time_mlp_dropout=0.3,
    ).cuda(device=dev)  #, bidirection=True

    epochs = 100
    warm_up_epochs = 10
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # criterion
    criterion = torch.nn.MSELoss()

    # other parameter
    score_index = 0

    min_val_loss = 10000
    max_spear_cor = 0
    patience = 10
    no_improve_epochs = 0

    model.train()

    for epoch_idx in range(epochs):
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
            train_pred.append(output.detach().data.cpu().numpy())
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
            no_improve_epochs = 0
            
            torch.save(model.state_dict(), "./fs800_result/checkpoint_pe.pth")
        else:
            no_improve_epochs += 1
        if spear > max_spear_cor:
            max_spear_cor = spear
        print("min validation loss: ", min_val_loss, " | max spear corr: ", max_spear_cor)
        print("checkpoint_pe")
        
        
        print(optimizer.param_groups[0]['lr'])
        # scheduler.step()
        if no_improve_epochs >= patience:
            print(f"Early stop triggered: val loss does not improve for {patience} epochs.")
            break

        
