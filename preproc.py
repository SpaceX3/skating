import os
import math
import argparse
import numpy as np
import torch
from torch import nn


def load_videoswin_backbone(device: str = "cuda:0", ckpt_dir: str = "./videomae-base-finetuned-ucf101"):
    """
    实际使用本地的 VideoMAE-base（finetuned UCF101）作为视频编码 backbone。
    通过 transformers 从本地目录加载，不访问网络。
    """
    try:
        from transformers import VideoMAEModel, AutoImageProcessor
    except ImportError:
        raise ImportError("需要安装 transformers 库以使用 VideoMAE，例如: pip install transformers")

    processor = AutoImageProcessor.from_pretrained(ckpt_dir, local_files_only=True)
    model = VideoMAEModel.from_pretrained(ckpt_dir, local_files_only=True)
    model.eval().to(device)
    return processor, model


def extract_videoswin_feature(backbone, video_tensor: torch.Tensor, device: str = "cuda:0") -> torch.Tensor:
    """
    使用 VideoMAE 对一个 clip 的多帧提特征。
    video_tensor: [T, C, H, W]
    返回: [D]，通常 D=768。
    """
    processor, model = backbone

    if video_tensor.dim() != 4:
        raise ValueError("video_tensor 期望维度为 [T, C, H, W]")

    # 转成 list[H,W,C]，让 VideoMAE 的 processor 处理
    frames = (video_tensor * 255.0).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()  # [T,H,W,C]

    from transformers import VideoMAEImageProcessor  # 仅类型提示用

    inputs = processor(list(frames), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)  # [1, C, T, H, W]

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        # 使用 CLS token 作为 clip 特征
        last_hidden = outputs.last_hidden_state  # [B, num_tokens, D]
        cls_feat = last_hidden[:, 0]  # [B, D]

    return cls_feat.squeeze(0).cpu()  # [D]


def read_video_clip(cap, start_sec: float, end_sec: float, target_fps: float = 8.0, size=(256, 256)):
    """
    从 cv2.VideoCapture 中按时间范围 [start_sec, end_sec] 抽帧，重采样到 target_fps，并 resize。
    返回: torch.Tensor [T, C, H, W]
    """
    import cv2

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))

    duration = max(1e-6, end_sec - start_sec)
    num_target = max(1, int(target_fps * duration))
    frame_indices = np.linspace(start_frame, end_frame - 1, num=num_target).astype(int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size)  # SwinV2 config.json 要求 256x256
        frame = torch.from_numpy(frame).float() / 255.0  # [H,W,3]
        frame = frame.permute(2, 0, 1)  # [3,H,W]
        frames.append(frame)

    if not frames:
        frames = [torch.zeros(3, size[1], size[0])]

    video_tensor = torch.stack(frames, dim=0)  # [T,3,H,W]
    return video_tensor


def process_one_video(model, video_path: str, out_path: str, device: str = "cuda:0", clip_len_sec: float = 5.0, overlap_ratio: float = 0.5):
    """
    使用 VST (Video Swin Transformer) 对单个视频做特征提取。

    - 每个 clip 长度为 clip_len_sec（默认 5s）
    - clip 之间重叠 overlap_ratio（默认 0.5，即 2.5s stride）
    - 输出特征 shape 设计为 [T_clip, 15, 768]，与原 Timesformer 特征在通道和 token 维度上兼容：
        - 时间维 T_clip 按 5s clip + 2.5s stride 滑动得到
        - 第 0 个 token 填入 VST 的 clip 特征，其余 token 置零
    """
    import cv2
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[preproc][warn] failed to open video: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames > 0 else 0.0
    if duration <= 0:
        print(f"[preproc][warn] invalid duration, skip: {video_path}")
        cap.release()
        return

    stride_sec = clip_len_sec * (1.0 - overlap_ratio)  # e.g. 5s -> 2.5s stride

    clip_feats = []
    t = 0.0
    while t < duration:
        start_sec = t
        end_sec = min(t + clip_len_sec, duration)
        video_tensor = read_video_clip(cap, start_sec, end_sec)
        feat = extract_videoswin_feature(model, video_tensor, device=device)  # [D] 或 [T,D]

        # 双保险：如果仍然是 [T,D]，再在这里做一次时间平均到 [D]
        if feat.dim() > 1:
            feat = feat.view(-1, feat.shape[-1]).mean(dim=0)

        # 使用最后一维作为特征维度
        D = feat.shape[-1]
        target_D = 768
        if D != target_D:
            # 简单线性映射至 768 维（仅用于保证形状一致；更合理做法是固定一个映射层并复用）
            proj = nn.Linear(D, target_D, bias=False)
            with torch.no_grad():
                feat = proj(feat.unsqueeze(0)).squeeze(0)

        token_feat = torch.zeros(15, target_D, dtype=torch.float32)
        token_feat[0] = feat
        clip_feats.append(token_feat.unsqueeze(0))  # [1,15,768]

        t += stride_sec

    cap.release()

    if not clip_feats:
        return

    clips = torch.cat(clip_feats, dim=0)  # [T_clip,15,768]
    np.save(out_path, clips.numpy().astype(np.float32))
    print(f"[preproc] saved feature: {out_path}, shape={clips.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../FS1000 Dataset/")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "both"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--clip_len", type=float, default=5.0)
    parser.add_argument("--overlap_ratio", type=float, default=0.5)
    parser.add_argument("--ckpt_dir", type=str, default="./videomae-base-finetuned-ucf101",
                        help="本地 VideoMAE-base 权重目录，包含 config.json / pytorch_model.bin 等")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model = load_videoswin_backbone(device=device, ckpt_dir=args.ckpt_dir)

    def process_split(split_name: str):
        txt_path = os.path.join(args.root, f"{split_name}_fs800.txt")
        out_dir = os.path.join(args.root, "VST_feature_fs800")
        os.makedirs(out_dir, exist_ok=True)

        with open(txt_path, "r") as f:
            lines = [line.strip().split() for line in f.readlines()]

        for i, data in enumerate(lines):
            data_index = data[0]
            video_path = os.path.join(args.root, "fs1000", f"{data_index}.mp4")
            out_path = os.path.join(out_dir, f"{data_index}.npy")
            if os.path.exists(out_path):
                continue
            process_one_video(
                model,
                video_path=video_path,
                out_path=out_path,
                device=device,
                clip_len_sec=args.clip_len,
                overlap_ratio=args.overlap_ratio,
            )
            if (i + 1) % 10 == 0:
                print(f"[preproc] {split_name}: {i+1}/{len(lines)} processed")

    if args.split in ("train", "both"):
        process_split("train")
    if args.split in ("val", "both"):
        process_split("val")


if __name__ == "__main__":
    main()


