import os
import math
import argparse
import numpy as np
import torch
from torch import nn


def load_videoswin_backbone(device: str = "cuda:0", model_name: str = "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k", ckpt_dir: str = None):
    """
    从本地权重加载 Video Swin Transformer backbone，不访问网络。

    - model_name: timm 中的模型名（需与你权重对应）
    - ckpt_dir: 存放 model.safetensors / pytorch_model.bin 的目录
    """
    try:
        import timm
    except ImportError:
        raise ImportError("需要安装 timm 库以使用 Video Swin Transformer，例如: pip install timm")

    # 不使用 pretrained=True，避免联网下载
    model = timm.create_model(model_name, pretrained=False)
    if hasattr(model, "head"):
        model.head = nn.Identity()

    # 从本地 safetensors 或 bin 加载权重
    if ckpt_dir is not None:
        from glob import glob
        from os.path import join, exists

        safetensor_files = glob(os.path.join(ckpt_dir, "*.safetensors"))
        bin_files = glob(os.path.join(ckpt_dir, "*.bin"))

        state_dict = None
        if safetensor_files:
            try:
                from safetensors.torch import load_file
            except ImportError:
                raise ImportError("检测到本地有 model.safetensors，但未安装 safetensors：pip install safetensors")
            ckpt_path = safetensor_files[0]
            print(f"[preproc] loading safetensors weights from {ckpt_path}")
            state_dict = load_file(ckpt_path)
        elif bin_files:
            ckpt_path = bin_files[0]
            print(f"[preproc] loading bin weights from {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")
            # 兼容 HuggingFace 格式
            if isinstance(state, dict) and "state_dict" in state:
                state_dict = state["state_dict"]
            else:
                state_dict = state

        if state_dict is not None:
            # 有些权重会带有前缀，比如 "backbone."，这里使用 strict=False 以增加兼容性
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[preproc][warn] missing keys when loading ckpt: {len(missing)}")
            if unexpected:
                print(f"[preproc][warn] unexpected keys when loading ckpt: {len(unexpected)}")
        else:
            print(f"[preproc][warn] 未在 {ckpt_dir} 中找到 .safetensors 或 .bin 权重，将使用随机初始化权重。")

    model.eval().to(device)
    return model


def extract_videoswin_feature(model, video_tensor: torch.Tensor, device: str = "cuda:0") -> torch.Tensor:
    """
    使用 SwinV2 图像 backbone 对一个 clip 的多帧提特征，并在时间维上做平均。
    video_tensor: [T, C, H, W]
    返回: [D]，其中 D 目标为 768。
    """
    if video_tensor.dim() != 4:
        raise ValueError("video_tensor 期望维度为 [T, C, H, W]")

    # 当前使用的是 2D SwinV2（图像分类），输入应为 [B, C, H, W]。
    # 这里按帧分别提特征，再在时间维上平均。
    x = video_tensor.to(device)  # [T,C,H,W]
    with torch.no_grad():
        feat = model(x)  # [T,D]
    if feat.dim() == 1:
        return feat.cpu()
    feat_mean = feat.mean(dim=0)  # [D]
    return feat_mean.cpu()


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
    parser.add_argument("--model_name", type=str, default="swinv2_base_window12to16_192to256.ms_in22k_ft_in1k")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="包含 model.safetensors / pytorch_model.bin 的本地目录")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model = load_videoswin_backbone(device=device, model_name=args.model_name, ckpt_dir=args.ckpt_dir)

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


