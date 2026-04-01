import os
import math
import argparse
import numpy as np
import torch
from torch import nn


def load_videoswin_backbone(device: str = "cuda:0", ckpt_dir: str = "./model/videomae-base-finetuned-ucf101"):
    """
    实际使用本地的 VideoMAE-base（finetuned UCF101）作为视频编码 backbone。
    通过 transformers 从本地目录加载，不访问网络。
    """
    try:
        from transformers import VideoMAEModel
    except ImportError:
        raise ImportError("需要安装 transformers 库以使用 VideoMAE，例如: pip install transformers")

    model = VideoMAEModel.from_pretrained(ckpt_dir, local_files_only=True)
    model.eval().to(device)
    num_frames = int(getattr(model.config, "num_frames", 16))
    image_size = int(getattr(model.config, "image_size", 224))
    # UCF101 finetuned VideoMAE 通常使用 ImageNet 归一化
    # pixel_values 形状为 [B, T, C, H, W]，因此 mean/std 形状 [1,1,3,1,1]
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1)
    return {
        "model": model,
        "num_frames": num_frames,
        "image_size": image_size,
        "mean": mean,
        "std": std,
    }


def _sample_frames(video_tensor: torch.Tensor, num_frames: int) -> torch.Tensor:
    """video_tensor: [T,C,H,W] -> [num_frames,C,H,W]"""
    T = video_tensor.shape[0]
    indices = np.linspace(0, max(T - 1, 0), num_frames).astype(int)
    return video_tensor[indices]


def load_video_frames_decord(video_path: str, image_size: int = 224):
    """
    使用 decord 一次性顺序解码整段视频，避免多次随机 seek。
    返回:
      frames: [T, C, H, W] in [0,1]
      fps: float (如果不可得则使用 25.0 作为近似)
    """
    try:
        import decord
        from decord import VideoReader, cpu
    except ImportError as e:
        raise ImportError("decord 未安装，请先安装：pip install decord") from e

    import torch.nn.functional as F

    vr = VideoReader(video_path, ctx=cpu(0))
    T_total = len(vr)

    # 尝试从 metadata 中获取 fps
    try:
        fps = float(vr.get_avg_fps())
        if fps <= 0:
            fps = 25.0
    except Exception:
        fps = 25.0

    # 读取所有帧并转成 torch.Tensor
    frames_nd = vr.get_batch(list(range(T_total)))  # [T,H,W,C], uint8
    frames = torch.from_numpy(frames_nd.asnumpy()).float() / 255.0  # [T,H,W,C] -> [0,1]
    frames = frames.permute(0, 3, 1, 2)  # [T,C,H,W]

    # 统一 resize
    frames = F.interpolate(
        frames,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return frames, fps


def _to_pixel_values(sampled: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, device: str) -> torch.Tensor:
    """sampled: [F,C,H,W] in [0,1] -> pixel_values [1,F,C,H,W] normalized"""
    x = sampled.float().unsqueeze(0)  # [1,F,C,H,W]
    x = x.to(device, non_blocking=True)
    x = (x - mean.to(device)) / std.to(device)
    return x


def extract_videoswin_feature(backbone, video_tensor: torch.Tensor, device: str = "cuda:0") -> torch.Tensor:
    """
    使用 VideoMAE 对一个 clip 的多帧提特征。
    video_tensor: [T, C, H, W]
    返回: [D]，通常 D=768。
    """
    model = backbone["model"]
    num_frames = backbone["num_frames"]
    mean = backbone["mean"]
    std = backbone["std"]

    if video_tensor.dim() != 4:
        raise ValueError("video_tensor 期望维度为 [T, C, H, W]")

    sampled = _sample_frames(video_tensor, num_frames=num_frames)
    pixel_values = _to_pixel_values(sampled, mean=mean, std=std, device=device)  # [1,C,T,H,W]

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        # 使用 CLS token 作为 clip 特征
        last_hidden = outputs.last_hidden_state  # [B, num_tokens, D]
        cls_feat = last_hidden[:, 0]  # [B, D]

    return cls_feat.squeeze(0).cpu()  # [D]


def read_video_clip(cap, start_sec: float, end_sec: float, target_fps: float = 8.0, size=(224, 224)):
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


def read_video_clip_decord(vr, fps: float, start_sec: float, end_sec: float, target_fps: float = 8.0, size=(224, 224)):
    """
    用 decord 从单个 clip 范围抽帧，避免整视频读入内存。
    返回: [T, C, H, W]
    """
    import torch.nn.functional as F

    total_frames = len(vr)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))

    duration = max(1e-6, end_sec - start_sec)
    num_target = max(1, int(target_fps * duration))
    frame_indices = np.linspace(start_frame, end_frame - 1, num=num_target).astype(int).tolist()

    nd = vr.get_batch(frame_indices)  # [T,H,W,C], uint8
    frames = torch.from_numpy(nd.asnumpy()).float() / 255.0
    frames = frames.permute(0, 3, 1, 2)  # [T,C,H,W]
    frames = F.interpolate(frames, size=size, mode="bilinear", align_corners=False)
    return frames


def _project_to_768(feat: torch.Tensor) -> torch.Tensor:
    """Deterministic mapping to 768 dims (no random projection)."""
    target_D = 768
    d = feat.shape[-1]
    if d == target_D:
        return feat
    if d > target_D:
        return feat[:target_D]
    out = torch.zeros(target_D, dtype=feat.dtype)
    out[:d] = feat
    return out


def process_one_video(model, video_path: str, out_path: str, device: str = "cuda:0", clip_len_sec: float = 5.0, overlap_ratio: float = 0.5, infer_batch_size: int = 16):
    """
    使用 VST (Video Swin Transformer) 对单个视频做特征提取。

    - 每个 clip 长度为 clip_len_sec（默认 5s）
    - clip 之间重叠 overlap_ratio（默认 0.5，即 2.5s stride）
    - 输出特征 shape 设计为 [T_clip, 15, 768]，与原 Timesformer 特征在通道和 token 维度上兼容：
        - 时间维 T_clip 按 5s clip + 2.5s stride 滑动得到
        - 第 0 个 token 填入 VST 的 clip 特征，其余 token 置零
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stride_sec = clip_len_sec * (1.0 - overlap_ratio)
    if stride_sec <= 0:
        stride_sec = clip_len_sec

    clip_tensors = []

    # 优先使用 decord 按 clip 抽帧；失败则回退到 cv2 按 clip 读取
    try:
        import decord
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 25.0
        if fps <= 0:
            fps = 25.0
        duration = len(vr) / fps if len(vr) > 0 else 0.0
        if duration <= 0:
            print(f"[preproc][warn] invalid duration, skip: {video_path}")
            return

        t = 0.0
        while t < duration:
            start_sec = t
            end_sec = min(t + clip_len_sec, duration)
            clip_tensor = read_video_clip_decord(
                vr,
                fps=fps,
                start_sec=start_sec,
                end_sec=end_sec,
                target_fps=8.0,
                size=(model["image_size"], model["image_size"]),
            )
            clip_tensors.append(clip_tensor)
            t += stride_sec
    except Exception as e:
        print(f"[preproc][warn] decord VideoReader failed for {video_path}, fallback to cv2: {e}")
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[preproc][warn] failed to open video: {video_path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if total_frames > 0 else 0.0

        clip_tensors = []
        t = 0.0
        while t < duration:
            start_sec = t
            end_sec = min(t + clip_len_sec, duration)
            video_tensor = read_video_clip(cap, start_sec, end_sec, size=(model["image_size"], model["image_size"]))
            clip_tensors.append(video_tensor)
            t += stride_sec
        cap.release()

    if not clip_tensors:
        return

    # 批量前向，提升 GPU 计算占用
    clip_feats = []
    num_frames = model["num_frames"]
    mean = model["mean"]
    std = model["std"]
    backbone = model["model"]
    with torch.no_grad():
        for s in range(0, len(clip_tensors), infer_batch_size):
            chunk = clip_tensors[s:s + infer_batch_size]
            sampled_list = [_sample_frames(v, num_frames=num_frames) for v in chunk]  # each [F,C,H,W]
            sampled_batch = torch.stack(sampled_list, dim=0)  # [B,F,C,H,W]
            # 归一化到 [0,1] 后再按 ImageNet 均值方差标准化
            sampled_batch = sampled_batch.to(device, non_blocking=True)
            sampled_batch = (sampled_batch - mean.to(device)) / std.to(device)  # [B,F,C,H,W]
            out = backbone(pixel_values=sampled_batch)
            cls = out.last_hidden_state[:, 0].detach().cpu()  # [B,D]
            clip_feats.append(cls)

    clip_feats = torch.cat(clip_feats, dim=0)  # [T_clip,D]
    clip_feats = torch.stack([_project_to_768(f) for f in clip_feats], dim=0)  # [T_clip,768]

    token = torch.zeros(clip_feats.shape[0], 15, 768, dtype=torch.float32)
    token[:, 0, :] = clip_feats
    clips = token  # [T_clip,15,768]
    np.save(out_path, clips.numpy().astype(np.float32))
    print(f"[preproc] saved feature: {out_path}, shape={clips.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../FS1000 Dataset/")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "both"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--clip_len", type=float, default=5.0)
    parser.add_argument("--overlap_ratio", type=float, default=0.5)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--ckpt_dir", type=str, default="./model/videomae-base-finetuned-ucf101",
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
                infer_batch_size=args.infer_batch_size,
            )
            if (i + 1) % 10 == 0:
                print(f"[preproc] {split_name}: {i+1}/{len(lines)} processed")

    if args.split in ("train", "both"):
        process_split("train")
    if args.split in ("val", "both"):
        process_split("val")


if __name__ == "__main__":
    main()


