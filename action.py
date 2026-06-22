import argparse
import csv
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


def _read_keyframe_rows(path):
    rows = []
    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rows.append(
                    {
                        "segment_idx": i,
                        "frame_index": int(float(line.split()[0])),
                        "time_sec": None,
                        "segment_start": None,
                        "segment_end": None,
                    }
                )
        return rows

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if "frame_index" not in row:
                continue
            rows.append(
                {
                    "segment_idx": int(float(row.get("segment_idx") or i)),
                    "frame_index": int(float(row["frame_index"])),
                    "time_sec": _optional_float(row.get("time_sec")),
                    "segment_start": _optional_float(row.get("segment_start")),
                    "segment_end": _optional_float(row.get("segment_end")),
                }
            )
    return rows


def _optional_float(value):
    if value is None or value == "":
        return None
    return float(value)


def _find_keyframe_file(keyframe_root, data_index, keyframe_csv_name):
    if not keyframe_root:
        return None

    candidates = [
        os.path.join(keyframe_root, data_index, keyframe_csv_name),
        os.path.join(keyframe_root, data_index, "selected_keyframes.csv"),
        os.path.join(keyframe_root, data_index, "selected_frame_indices.txt"),
        os.path.join(keyframe_root, f"{data_index}_{keyframe_csv_name}"),
        os.path.join(keyframe_root, f"{data_index}.csv"),
        os.path.join(keyframe_root, keyframe_csv_name),
        os.path.join(keyframe_root, "selected_keyframes.csv"),
        os.path.join(keyframe_root, "selected_frame_indices.txt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _map_keyframes_to_tdyn(keyframe_rows, T_dyn, total_frames, fps):
    if not keyframe_rows:
        return None

    raw_indices = [int(row["frame_index"]) for row in keyframe_rows]
    if len(raw_indices) == T_dyn:
        return [max(0, min(idx, total_frames - 1)) for idx in raw_indices]

    duration = total_frames / fps if fps and fps > 0 else float(T_dyn)
    mapped = []
    for i in range(T_dyn):
        center_time = (i + 0.5) * duration / max(T_dyn, 1)
        containing = [
            row
            for row in keyframe_rows
            if row["segment_start"] is not None
            and row["segment_end"] is not None
            and row["segment_start"] <= center_time < row["segment_end"]
        ]
        if containing:
            chosen = containing[0]
        else:
            chosen = min(
                keyframe_rows,
                key=lambda row: abs(
                    (row["time_sec"] if row["time_sec"] is not None else row["frame_index"] / max(fps, 1e-6))
                    - center_time
                ),
            )
        idx = int(chosen["frame_index"])
        mapped.append(max(0, min(idx, total_frames - 1)))
    return mapped


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
    def extract(self, video_path, T_dyn, is_train, keyframe_rows=None):
        import cv2
        from PIL import Image

        self._lazy_init()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = T_dyn
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

        frame_indices = _map_keyframes_to_tdyn(keyframe_rows, T_dyn, total_frames, fps)
        if frame_indices is None:
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


def precompute_split(
    root_path,
    split_name,
    extractor,
    cache_dir,
    cache_prefix,
    failed_log_path,
    keyframe_root=None,
    keyframe_csv_name="selected_frame_indices.csv",
    require_keyframes=False,
    only_data_index=None,
    limit=None,
):
    rows = _read_split(root_path, split_name)
    os.makedirs(cache_dir, exist_ok=True)
    total = len(rows)
    print(f"[action] start split={split_name}, total={total}")

    processed = 0
    for i, row in enumerate(rows):
        data_index = row[0]
        if only_data_index is not None and data_index != only_data_index:
            continue
        if limit is not None and processed >= limit:
            break

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

        keyframe_rows = None
        keyframe_file = _find_keyframe_file(keyframe_root, data_index, keyframe_csv_name)
        if keyframe_file is not None:
            keyframe_rows = _read_keyframe_rows(keyframe_file)
            print(f"[action] {data_index}: using keyframes from {keyframe_file} ({len(keyframe_rows)} rows)")
            if len(keyframe_rows) != T_dyn:
                print(
                    f"[action][warn] {data_index}: keyframe rows ({len(keyframe_rows)}) != T_dyn ({T_dyn}); "
                    "mapping selected frames to dynamic timesteps. Use select_keyframes.py --num-segments "
                    "or --feature-path to produce one keyframe per overlapping dynamic window."
                )
        elif require_keyframes:
            raise FileNotFoundError(f"Missing keyframe file for {data_index} under {keyframe_root}")
        elif keyframe_root:
            print(f"[action][warn] {data_index}: keyframe file missing; fallback to original frame sampling")

        static_feat = extractor.extract(
            mp4_path,
            T_dyn=T_dyn,
            is_train=(split_name == "train"),
            keyframe_rows=keyframe_rows,
        )
        if static_feat is None:
            print(f"[action][warn] failed to open video, skip: {mp4_path}")
            _append_failed_video(failed_log_path, data_index, mp4_path)
            continue
        if static_feat.shape != (T_dyn, extractor.static_in_dim):
            raise ValueError(f"Unexpected static feature shape for {data_index}: {static_feat.shape}")
        tmp_path = out_path + f".tmp.{os.getpid()}.{random.randint(0, 1_000_000)}.npy"
        np.save(tmp_path, static_feat)
        os.replace(tmp_path, out_path)
        processed += 1
        print(f"[action] saved {out_path}, shape={static_feat.shape}")

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"[action] {split_name}: {i + 1}/{total}")
    print(f"[action] done split={split_name}, processed={processed}")


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
    parser.add_argument("--keyframe_root", type=str, default=None, help="Directory containing selected_frame_indices.csv outputs.")
    parser.add_argument("--keyframe_csv_name", type=str, default="selected_frame_indices.csv")
    parser.add_argument("--require_keyframes", action="store_true")
    parser.add_argument("--only_data_index", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--proxy", type=str, default=None, help="Optional HTTP/HTTPS proxy, e.g. http://127.0.0.1:6518")
    args = parser.parse_args()

    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
        os.environ["ALL_PROXY"] = args.proxy

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
        precompute_split(
            args.root_path,
            "train",
            extractor,
            cache_dir,
            args.cache_prefix,
            failed_log_path,
            keyframe_root=args.keyframe_root,
            keyframe_csv_name=args.keyframe_csv_name,
            require_keyframes=args.require_keyframes,
            only_data_index=args.only_data_index,
            limit=args.limit,
        )
    if args.split in ("val", "all"):
        precompute_split(
            args.root_path,
            "val",
            extractor,
            cache_dir,
            args.cache_prefix,
            failed_log_path,
            keyframe_root=args.keyframe_root,
            keyframe_csv_name=args.keyframe_csv_name,
            require_keyframes=args.require_keyframes,
            only_data_index=args.only_data_index,
            limit=args.limit,
        )

    print("[action] static feature precompute finished.")
