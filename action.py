import argparse
import os
import random
import numpy as np
import torch
from glob import glob


def _safe_read_video_frame(cap, frame_idx):
    import cv2

    if frame_idx < 0:
        frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        return None
    return frame_bgr


class ResNet50StaticFeatureExtractor:
    def __init__(self, device="cuda:0", static_in_dim=2048, infer_batch_size=64, use_amp=True):
        self.device = device
        self.static_in_dim = static_in_dim
        self.infer_batch_size = infer_batch_size
        self.use_amp = use_amp
        self._resnet_features = None
        self._preprocess = None

    def _lazy_init(self):
        if self._resnet_features is not None:
            return
        from torchvision import models
        from torchvision.models import ResNet50_Weights

        weights = ResNet50_Weights.IMAGENET1K_V2
        self._preprocess = weights.transforms()
        resnet = models.resnet50(weights=weights)
        self._resnet_features = torch.nn.Sequential(*list(resnet.children())[:-1]).to(self.device).eval()

    @torch.no_grad()
    def extract(self, video_path, T_dyn, is_train):
        import cv2
        from PIL import Image

        self._lazy_init()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = T_dyn

        frame_indices = []
        for i in range(T_dyn):
            start = int(i * total_frames / T_dyn)
            end = int((i + 1) * total_frames / T_dyn) - 1
            if end < start:
                end = start
            start = max(0, min(start, total_frames - 1))
            end = max(0, min(end, total_frames - 1))
            idx = random.randint(start, end) if is_train else (start + end) // 2
            frame_indices.append(idx)

        unique_indices = sorted(set(frame_indices))
        idx_to_feat = {}

        # Collect decoded+preprocessed frames first, then run batched GPU inference.
        frame_tensors = []
        kept_indices = []
        for idx in unique_indices:
            frame_bgr = _safe_read_video_frame(cap, idx)
            if frame_bgr is None:
                idx_to_feat[idx] = np.zeros((self.static_in_dim,), dtype=np.float32)
                continue
            frame_rgb = frame_bgr[:, :, ::-1]
            img = Image.fromarray(frame_rgb)
            x = self._preprocess(img)  # [3, H, W]
            frame_tensors.append(x)
            kept_indices.append(idx)

        if len(frame_tensors) > 0:
            with torch.no_grad():
                for start in range(0, len(frame_tensors), self.infer_batch_size):
                    end = min(start + self.infer_batch_size, len(frame_tensors))
                    batch = torch.stack(frame_tensors[start:end], dim=0).to(self.device, non_blocking=True)
                    if self.use_amp and str(self.device).startswith("cuda"):
                        with torch.cuda.amp.autocast():
                            emb = self._resnet_features(batch).flatten(1)
                    else:
                        emb = self._resnet_features(batch).flatten(1)
                    emb = emb.detach().cpu().numpy().astype(np.float32)
                    for i, idx in enumerate(kept_indices[start:end]):
                        idx_to_feat[idx] = emb[i]

        cap.release()
        return np.stack([idx_to_feat[idx] for idx in frame_indices], axis=0).astype(np.float32)


def _read_split(root_path, split_name):
    split_file = os.path.join(root_path, f"{split_name}_fs800.txt")
    with open(split_file, "r") as f:
        rows = [line.strip().split() for line in f.readlines()]
    return rows


def _cache_path(cache_dir, prefix, data_index, T_dyn):
    return os.path.join(cache_dir, f"{prefix}_{data_index}_T{T_dyn}.npy")


def _already_precomputed_any(cache_dir, cache_prefix, data_index):
    pattern = os.path.join(cache_dir, f"{cache_prefix}_{data_index}_T*.npy")
    return len(glob(pattern)) > 0


def _append_failed_video(failed_log_path, data_index, video_path):
    with open(failed_log_path, "a") as f:
        f.write(f"{data_index} {video_path}\n")


def precompute_split(root_path, split_name, extractor, cache_dir, cache_prefix, failed_log_path):
    rows = _read_split(root_path, split_name)
    os.makedirs(cache_dir, exist_ok=True)
    total = len(rows)
    print(f"[action] start split={split_name}, total={total}")

    for i, row in enumerate(rows):
        data_index = row[0]
        audio_path = os.path.join(root_path, "new feature", "ast_feature_fs1000_new", data_index + ".npy")
        video_path = os.path.join(root_path, "Timesformer_output_feature_fs800", data_index + ".npy")
        mp4_path = os.path.join(root_path, "fs1000", f"{data_index}.mp4")

        audio_feat = np.load(audio_path)
        video_feat = np.load(video_path)
        T_dyn = min(audio_feat.shape[0], video_feat.shape[0])
        out_path = _cache_path(cache_dir, cache_prefix, data_index, T_dyn)
        # Skip if exact cache exists or this video has been precomputed before.
        if os.path.exists(out_path) or _already_precomputed_any(cache_dir, cache_prefix, data_index):
            continue

        static_feat = extractor.extract(mp4_path, T_dyn=T_dyn, is_train=(split_name == "train"))
        if static_feat is None:
            print(f"[action][warn] failed to open video, skip: {mp4_path}")
            _append_failed_video(failed_log_path, data_index, mp4_path)
            continue
        tmp_path = out_path + f".tmp.{os.getpid()}.{random.randint(0, 1_000_000)}.npy"
        np.save(tmp_path, static_feat)
        os.replace(tmp_path, out_path)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"[action] {split_name}: {i + 1}/{total}")
    print(f"[action] done split={split_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="../FS1000 Dataset/")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cache_dir_name", type=str, default="static_resnet50_cache")
    parser.add_argument("--cache_prefix", type=str, default="static_resnet50")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "all"])
    parser.add_argument("--infer_batch_size", type=int, default=64)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--failed_log_name", type=str, default="static_resnet50_failed_videos.txt")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    extractor = ResNet50StaticFeatureExtractor(
        device=device,
        static_in_dim=2048,
        infer_batch_size=args.infer_batch_size,
        use_amp=(not args.disable_amp),
    )
    cache_dir = os.path.join(args.root_path, args.cache_dir_name)
    failed_log_path = os.path.join(args.root_path, args.failed_log_name)

    if args.split in ("train", "all"):
        precompute_split(args.root_path, "train", extractor, cache_dir, args.cache_prefix, failed_log_path)
    if args.split in ("val", "all"):
        precompute_split(args.root_path, "val", extractor, cache_dir, args.cache_prefix, failed_log_path)

    print("[action] static feature precompute finished.")
