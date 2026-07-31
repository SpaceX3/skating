import argparse
import hashlib
import os
import random
import time
from glob import glob

import numpy as np
import torch


def read_split(root_path, split_name):
    split_file = os.path.join(root_path, f"{split_name}_fs800.txt")
    with open(split_file, "r") as f:
        return [line.strip().split() for line in f if line.strip()]


def dynamic_feature_paths(root_path, data_index):
    audio_path = os.path.join(
        root_path, "new feature", "ast_feature_fs1000_new", data_index + ".npy"
    )
    video_path = os.path.join(
        root_path, "Timesformer_output_feature_fs800", data_index + ".npy"
    )
    return audio_path, video_path


def read_t_dyn(root_path, data_index):
    audio_path, video_path = dynamic_feature_paths(root_path, data_index)
    audio_feat = np.load(audio_path, mmap_mode="r")
    video_feat = np.load(video_path, mmap_mode="r")
    return int(min(audio_feat.shape[0], video_feat.shape[0]))


def cache_path(cache_dir, cache_prefix, data_index, t_dyn):
    return os.path.join(cache_dir, f"{cache_prefix}_{data_index}_T{t_dyn}.npy")


def times_path(cache_dir, cache_prefix, data_index, t_dyn):
    return os.path.join(
        cache_dir, f"{cache_prefix}_{data_index}_T{t_dyn}.times.npy"
    )


def frame_cache_path(cache_dir, cache_prefix, data_index, t_dyn):
    return cache_path(cache_dir, cache_prefix + "_frames", data_index, t_dyn)


def frame_times_path(cache_dir, cache_prefix, data_index, t_dyn):
    return times_path(cache_dir, cache_prefix + "_frames", data_index, t_dyn)


def atomic_save(path, values):
    tmp_path = path + f".tmp.{os.getpid()}.{random.randint(0, 1_000_000)}.npy"
    np.save(tmp_path, values)
    os.replace(tmp_path, path)


def window_times_array(windows, sample_first_sec):
    return np.asarray(
        [
            [start, end, start, min(start + sample_first_sec, end)]
            for _, start, end in windows
        ],
        dtype=np.float32,
    )


def already_precomputed_any(cache_dir, cache_prefix, data_index):
    pattern = os.path.join(cache_dir, f"{cache_prefix}_{data_index}_T*.npy")
    return any(not path.endswith(".times.npy") for path in glob(pattern))


def append_failed_video(failed_log_path, data_index, video_path, reason):
    os.makedirs(os.path.dirname(failed_log_path), exist_ok=True)
    with open(failed_log_path, "a") as f:
        f.write(f"{data_index} {video_path} {reason}\n")


def default_dinov2_hub_dir():
    hub_dir = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
    if os.path.exists(os.path.join(hub_dir, "hubconf.py")):
        return hub_dir
    return None


def build_segment_windows(duration, clip_len, num_segments):
    """Match keyframe_selector feature-path mode: use feature T as segment count."""
    if clip_len <= 0:
        raise ValueError("clip_len must be positive")
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")
    if duration <= 0:
        return []

    if num_segments == 1:
        starts = [0.0]
    elif duration <= clip_len:
        starts = [0.0 for _ in range(num_segments)]
    else:
        stride = (duration - clip_len) / float(num_segments - 1)
        starts = [i * stride for i in range(num_segments)]

    windows = []
    for idx, start in enumerate(starts):
        if duration > clip_len:
            start = min(start, duration - clip_len)
        start = max(0.0, min(float(start), max(duration - 1e-6, 0.0)))
        end = min(start + clip_len, duration)
        if end <= start:
            end = min(duration, start + 1e-6)
        windows.append((idx, start, end))
    return windows


def frame_range_for_time(fps, total_frames, start_sec, end_sec):
    start_idx = int(np.ceil(start_sec * fps))
    end_idx = int(np.ceil(end_sec * fps)) - 1
    start_idx = max(0, min(start_idx, total_frames - 1))
    end_idx = max(0, min(end_idx, total_frames - 1))
    if end_idx < start_idx:
        end_idx = start_idx
    return start_idx, end_idx


def sample_frame_indices(
    rng,
    fps,
    total_frames,
    segment_start,
    segment_end,
    sample_first_sec,
    frames_per_second,
):
    segment_duration = max(0.0, segment_end - segment_start)
    sample_count = int(round(sample_first_sec * frames_per_second))
    sample_count = max(sample_count, 1)

    if segment_duration <= 0:
        return [0 for _ in range(sample_count)]

    if segment_duration < sample_first_sec:
        start_idx, end_idx = frame_range_for_time(
            fps, total_frames, segment_start, segment_end
        )
        candidates = list(range(start_idx, end_idx + 1))
        if len(candidates) >= sample_count:
            return sorted(rng.sample(candidates, sample_count))
        return sorted(rng.choices(candidates, k=sample_count))

    indices = []
    full_seconds = int(np.floor(sample_first_sec))
    for sec_idx in range(full_seconds):
        sec_start = segment_start + float(sec_idx)
        sec_end = min(sec_start + 1.0, segment_start + sample_first_sec, segment_end)
        start_idx, end_idx = frame_range_for_time(fps, total_frames, sec_start, sec_end)
        candidates = list(range(start_idx, end_idx + 1))
        if len(candidates) >= frames_per_second:
            indices.extend(rng.sample(candidates, frames_per_second))
        else:
            indices.extend(rng.choices(candidates, k=frames_per_second))

    while len(indices) < sample_count:
        sec_start = segment_start
        sec_end = min(segment_start + sample_first_sec, segment_end)
        start_idx, end_idx = frame_range_for_time(fps, total_frames, sec_start, sec_end)
        candidates = list(range(start_idx, end_idx + 1))
        indices.extend(rng.sample(candidates, min(len(candidates), sample_count - len(indices))))
    return sorted(indices)


def stable_video_seed(base_seed, data_index):
    digest = hashlib.md5(data_index.encode("utf-8")).hexdigest()
    return int(base_seed) + int(digest[:8], 16)


class DinoV2StaticFeatureExtractor:
    def __init__(
        self,
        device="cuda:0",
        model_name="dinov2_vitl14",
        output_dim=None,
        image_size=224,
        infer_batch_size=16,
        decode_batch_size=128,
        use_amp=True,
        clip_len=5.0,
        sample_first_sec=2.0,
        frames_per_second=2,
        hub_dir=None,
    ):
        self.device = device
        self.model_name = model_name
        self.expected_output_dim = (
            None if output_dim is None else int(output_dim)
        )
        self.image_size = int(image_size)
        self.infer_batch_size = int(infer_batch_size)
        self.decode_batch_size = int(decode_batch_size)
        self.use_amp = bool(use_amp)
        self.clip_len = float(clip_len)
        self.sample_first_sec = float(sample_first_sec)
        self.frames_per_second = int(frames_per_second)
        self.hub_dir = hub_dir
        self.feature_dim = None
        self.static_in_dim = None
        self.model = None
        self.preprocess = None
        self._logged_feature_shapes = False

    def lazy_init(self):
        if self.model is not None:
            return
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        load_start = time.time()
        hub_dir = self.hub_dir or default_dinov2_hub_dir()
        if hub_dir:
            print(f"[action] loading DINOv2 from local hub: {hub_dir}", flush=True)
            self.model = torch.hub.load(
                hub_dir,
                self.model_name,
                source="local",
                pretrained=True,
            )
        else:
            print("[action] loading DINOv2 from torch hub remote", flush=True)
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                self.model_name,
                pretrained=True,
                trust_repo=True,
            )
        self.model.eval().to(self.device)
        feature_dim = getattr(self.model, "embed_dim", None)
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError(
                f"DINOv2 model {self.model_name!r} does not expose a valid embed_dim"
            )
        self.feature_dim = feature_dim
        self.static_in_dim = 2 * feature_dim
        if (
            self.expected_output_dim is not None
            and self.expected_output_dim != self.static_in_dim
        ):
            raise ValueError(
                f"--output_dim={self.expected_output_dim} does not match the fused "
                f"CLS + Patch Mean dimension {self.static_in_dim} for "
                f"{self.model_name} (token dimension {self.feature_dim})"
            )
        print(
            f"[action] DINOv2 ready on {self.device} "
            f"({time.time() - load_start:.1f}s), token_dim={self.feature_dim}, "
            f"fused_dim={self.static_in_dim}",
            flush=True,
        )
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    self.image_size,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def forward_features(self, batch):
        features = self.model.forward_features(batch)
        if not isinstance(features, dict):
            raise TypeError(
                "DINOv2 forward_features must return a dictionary containing "
                "x_norm_clstoken and x_norm_patchtokens"
            )

        cls = features.get("x_norm_clstoken")
        patches = features.get("x_norm_patchtokens")
        if cls is None or patches is None:
            raise KeyError(
                "DINOv2 forward_features output is missing x_norm_clstoken or "
                "x_norm_patchtokens"
            )
        if cls.ndim != 2:
            raise ValueError(f"Expected CLS shape [B, D], got {tuple(cls.shape)}")
        if patches.ndim != 3:
            raise ValueError(
                f"Expected patch-token shape [B, N, D], got {tuple(patches.shape)}"
            )
        if patches.shape[1] == 0:
            raise ValueError("DINOv2 returned no patch tokens")
        if cls.shape[0] != patches.shape[0] or cls.shape[-1] != patches.shape[-1]:
            raise ValueError(
                "CLS and patch-token batch/feature dimensions do not match: "
                f"CLS={tuple(cls.shape)}, patches={tuple(patches.shape)}"
            )
        if self.feature_dim is not None and cls.shape[-1] != self.feature_dim:
            raise ValueError(
                f"DINOv2 token dimension changed from embed_dim={self.feature_dim} "
                f"to {cls.shape[-1]}"
            )

        patch_mean = patches.mean(dim=1)
        fused = torch.cat([cls, patch_mean], dim=-1)
        if not self._logged_feature_shapes:
            print(
                "[action] feature shapes: "
                f"cls={tuple(cls.shape)}, patches={tuple(patches.shape)}, "
                f"patch_mean={tuple(patch_mean.shape)}, fused={tuple(fused.shape)}",
                flush=True,
            )
            self._logged_feature_shapes = True
        return fused

    @torch.no_grad()
    def encode_frames(self, frames_rgb):
        self.lazy_init()
        if len(frames_rgb) == 0:
            return np.zeros((0, self.static_in_dim), dtype=np.float32)

        feats = []
        for start in range(0, len(frames_rgb), self.infer_batch_size):
            end = min(start + self.infer_batch_size, len(frames_rgb))
            tensors = [
                self.preprocess(np.ascontiguousarray(frame))
                for frame in frames_rgb[start:end]
            ]
            batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)
            if self.use_amp and str(self.device).startswith("cuda"):
                with torch.cuda.amp.autocast():
                    feat = self.forward_features(batch)
            else:
                feat = self.forward_features(batch)
            feats.append(feat.detach().float().cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32)

    def extract(self, video_path, t_dyn, rng):
        from decord import VideoReader, cpu

        self.lazy_init()
        try:
            video_reader = VideoReader(video_path, ctx=cpu(0))
        except Exception as exc:
            print(f"[action][warn] failed to open video: {video_path} ({exc})")
            return None

        total_frames = len(video_reader)
        if total_frames <= 0:
            print(f"[action][warn] empty video: {video_path}")
            return None

        fps = float(video_reader.get_avg_fps() or 25.0)
        duration = float(total_frames) / max(fps, 1e-6)
        windows = build_segment_windows(duration, self.clip_len, t_dyn)
        if len(windows) != t_dyn:
            raise ValueError(
                f"segment count mismatch for {video_path}: windows={len(windows)} T_dyn={t_dyn}"
            )

        per_segment_indices = [
            sample_frame_indices(
                rng=rng,
                fps=fps,
                total_frames=total_frames,
                segment_start=seg_start,
                segment_end=seg_end,
                sample_first_sec=self.sample_first_sec,
                frames_per_second=self.frames_per_second,
            )
            for _, seg_start, seg_end in windows
        ]

        unique_indices = sorted({idx for indices in per_segment_indices for idx in indices})
        print(
            f"[action] decode/infer frames: unique={len(unique_indices)}, "
            f"segments={t_dyn}, fps={fps:.3f}",
            flush=True,
        )
        idx_to_feat = {}
        for start in range(0, len(unique_indices), self.decode_batch_size):
            batch_indices = unique_indices[start : start + self.decode_batch_size]
            frames = video_reader.get_batch(batch_indices).asnumpy()
            feats = self.encode_frames(frames)
            for idx, feat in zip(batch_indices, feats):
                idx_to_feat[int(idx)] = feat

        segment_feats = []
        segment_frame_feats = []
        segment_frame_times = []
        for indices in per_segment_indices:
            frame_feats = np.stack([idx_to_feat[int(idx)] for idx in indices], axis=0)
            segment_feats.append(frame_feats.mean(axis=0))
            segment_frame_feats.append(frame_feats)
            segment_frame_times.append(
                np.asarray(indices, dtype=np.float32) / max(fps, 1e-6)
            )

        return (
            np.stack(segment_feats, axis=0).astype(np.float32),
            window_times_array(windows, self.sample_first_sec),
            np.stack(segment_frame_feats, axis=0).astype(np.float16),
            np.stack(segment_frame_times, axis=0).astype(np.float32),
        )


def precompute_split(
    root_path,
    split_name,
    extractor,
    cache_dir,
    cache_prefix,
    failed_log_path,
    only_data_index=None,
    limit=None,
    overwrite=False,
    seed=2026,
    save_frame_sequences=False,
):
    rows = read_split(root_path, split_name)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[action] start split={split_name}, total={len(rows)}", flush=True)

    processed = 0
    skipped_existing = 0
    for row_idx, row in enumerate(rows):
        data_index = row[0]
        if only_data_index is not None and data_index != only_data_index:
            continue
        if limit is not None and processed >= limit:
            break

        video_path = os.path.join(root_path, "fs1000", f"{data_index}.mp4")
        try:
            t_dyn = read_t_dyn(root_path, data_index)
        except Exception as exc:
            print(f"[action][warn] failed to read T_dyn for {data_index}: {exc}", flush=True)
            append_failed_video(failed_log_path, data_index, video_path, f"tdyn_error:{exc}")
            continue

        out_path = cache_path(cache_dir, cache_prefix, data_index, t_dyn)
        out_times_path = times_path(cache_dir, cache_prefix, data_index, t_dyn)
        out_frame_path = frame_cache_path(
            cache_dir, cache_prefix, data_index, t_dyn
        )
        out_frame_times_path = frame_times_path(
            cache_dir, cache_prefix, data_index, t_dyn
        )
        frame_cache_ready = (
            os.path.exists(out_frame_path)
            and os.path.exists(out_frame_times_path)
        )
        if not overwrite and (
            os.path.exists(out_path)
            or already_precomputed_any(cache_dir, cache_prefix, data_index)
        ) and os.path.exists(out_times_path) and (
            not save_frame_sequences or frame_cache_ready
        ):
            skipped_existing += 1
            continue

        video_start = time.time()
        print(
            f"[action] processing {split_name} {row_idx + 1}/{len(rows)}: "
            f"{data_index}, T_dyn={t_dyn}",
            flush=True,
        )
        rng = random.Random(stable_video_seed(seed, data_index))
        extracted = extractor.extract(video_path, t_dyn=t_dyn, rng=rng)
        if extracted is None:
            append_failed_video(failed_log_path, data_index, video_path, "video_error")
            continue
        static_feat, static_times, frame_feat, frame_times = extracted
        if static_feat.shape != (t_dyn, extractor.static_in_dim):
            raise ValueError(
                f"Unexpected static feature shape for {data_index}: "
                f"{static_feat.shape}, expected {(t_dyn, extractor.static_in_dim)}"
            )

        expected_frames = int(
            round(extractor.sample_first_sec * extractor.frames_per_second)
        )
        expected_frame_shape = (t_dyn, max(expected_frames, 1), extractor.static_in_dim)
        if frame_feat.shape != expected_frame_shape:
            raise ValueError(
                f"Unexpected frame feature shape for {data_index}: "
                f"{frame_feat.shape}, expected {expected_frame_shape}"
            )
        if frame_times.shape != frame_feat.shape[:2]:
            raise ValueError(
                f"Frame timestamps do not match frame features for {data_index}: "
                f"{frame_times.shape} versus {frame_feat.shape[:2]}"
            )

        atomic_save(out_path, static_feat)
        atomic_save(out_times_path, static_times)
        if save_frame_sequences:
            atomic_save(out_frame_path, frame_feat)
            atomic_save(out_frame_times_path, frame_times)
        processed += 1
        print(
            f"[action] saved {out_path}, shape={static_feat.shape}, "
            f"time={time.time() - video_start:.1f}s",
            flush=True,
        )

        if (row_idx + 1) % 50 == 0 or (row_idx + 1) == len(rows):
            print(f"[action] {split_name}: {row_idx + 1}/{len(rows)}", flush=True)

    print(
        f"[action] done split={split_name}, processed={processed}, "
        f"existing={skipped_existing}",
        flush=True,
    )
    return processed


def finefs_rows(root_path):
    annotation_dir = os.path.join(root_path, "annotation")
    return [
        os.path.splitext(name)[0]
        for name in sorted(os.listdir(annotation_dir), key=lambda value: int(os.path.splitext(value)[0]))
        if name.endswith(".json")
    ]


def finefs_segment_count(duration, clip_len, stride):
    if duration <= clip_len:
        return 1
    return int(np.floor((duration - clip_len) / stride)) + 1


def precompute_finefs(
    root_path,
    extractor,
    cache_dir,
    cache_prefix,
    failed_log_path,
    only_data_index=None,
    limit=None,
    overwrite=False,
    seed=2026,
    stride=2.0,
    save_frame_sequences=False,
):
    from decord import VideoReader, cpu

    rows = finefs_rows(root_path)
    os.makedirs(cache_dir, exist_ok=True)
    processed = 0
    skipped_existing = 0
    for row_idx, data_index in enumerate(rows):
        if only_data_index is not None and data_index != only_data_index:
            continue
        if limit is not None and processed >= limit:
            break
        video_path = os.path.join(root_path, "video", data_index + ".mp4")
        try:
            reader = VideoReader(video_path, ctx=cpu(0))
            duration = float(len(reader)) / max(float(reader.get_avg_fps()), 1e-6)
            t_dyn = finefs_segment_count(duration, extractor.clip_len, stride)
        except Exception as exc:
            append_failed_video(
                failed_log_path, data_index, video_path, f"metadata_error:{exc}"
            )
            continue

        out_path = cache_path(cache_dir, cache_prefix, data_index, t_dyn)
        out_times_path = times_path(cache_dir, cache_prefix, data_index, t_dyn)
        out_frame_path = frame_cache_path(
            cache_dir, cache_prefix, data_index, t_dyn
        )
        out_frame_times_path = frame_times_path(
            cache_dir, cache_prefix, data_index, t_dyn
        )
        frame_cache_ready = (
            os.path.exists(out_frame_path)
            and os.path.exists(out_frame_times_path)
        )
        if (
            not overwrite
            and os.path.exists(out_path)
            and os.path.exists(out_times_path)
            and (not save_frame_sequences or frame_cache_ready)
        ):
            skipped_existing += 1
            continue
        print(
            f"[action] FineFS {row_idx + 1}/{len(rows)}: {data_index}, "
            f"duration={duration:.2f}, T={t_dyn}",
            flush=True,
        )
        rng = random.Random(stable_video_seed(seed, "finefs_" + data_index))
        extracted = extractor.extract(video_path, t_dyn=t_dyn, rng=rng)
        if extracted is None:
            append_failed_video(failed_log_path, data_index, video_path, "video_error")
            continue
        static_feat, static_times, frame_feat, frame_times = extracted
        if frame_feat.shape[:2] != frame_times.shape:
            raise ValueError(
                f"Frame timestamps do not match frame features for {data_index}: "
                f"{frame_times.shape} versus {frame_feat.shape[:2]}"
            )
        atomic_save(out_path, static_feat)
        atomic_save(out_times_path, static_times)
        if save_frame_sequences:
            atomic_save(out_frame_path, frame_feat)
            atomic_save(out_frame_times_path, frame_times)
        processed += 1
    print(
        f"[action] FineFS done: processed={processed}, existing={skipped_existing}",
        flush=True,
    )


def build_fs1000_time_sidecars(root_path, cache_dir, cache_prefix, clip_len, sample_first_sec):
    from decord import VideoReader, cpu

    written = 0
    for split_name in ("train", "val"):
        for row in read_split(root_path, split_name):
            data_index = row[0]
            t_dyn = read_t_dyn(root_path, data_index)
            feature_path = cache_path(cache_dir, cache_prefix, data_index, t_dyn)
            if not os.path.exists(feature_path):
                raise FileNotFoundError(feature_path)
            out_times_path = times_path(cache_dir, cache_prefix, data_index, t_dyn)
            if os.path.exists(out_times_path):
                continue
            video_path = os.path.join(root_path, "fs1000", data_index + ".mp4")
            reader = VideoReader(video_path, ctx=cpu(0))
            duration = float(len(reader)) / max(float(reader.get_avg_fps()), 1e-6)
            windows = build_segment_windows(duration, clip_len, t_dyn)
            np.save(out_times_path, window_times_array(windows, sample_first_sec))
            written += 1
    print(f"[action] wrote {written} FS1000 time sidecars", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="../FS1000 Dataset/")
    parser.add_argument(
        "--dataset_mode", choices=["fs1000", "finefs"], default="fs1000"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--cache_dir_name",
        type=str,
        default="static_dinov2_cls_patch_mean_cache",
    )
    parser.add_argument(
        "--cache_prefix",
        type=str,
        default="static_dinov2_cls_patch_mean",
    )
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "all"])
    parser.add_argument("--model_name", type=str, default="dinov2_vitl14")
    parser.add_argument(
        "--output_dim",
        type=int,
        default=None,
        help=(
            "Optional compatibility check for the fused output dimension. The actual "
            "dimension is always inferred as 2 * backbone.embed_dim."
        ),
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--decode_batch_size", type=int, default=128)
    parser.add_argument("--clip_len", type=float, default=5.0)
    parser.add_argument("--sample_first_sec", type=float, default=2.0)
    parser.add_argument("--frames_per_second", type=int, default=2)
    parser.add_argument(
        "--save_frame_sequences",
        action="store_true",
        help=(
            "Save ordered per-frame DINO sidecars in addition to the existing "
            "per-window mean cache."
        ),
    )
    parser.add_argument("--hub_dir", type=str, default=None)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument(
        "--failed_log_name",
        type=str,
        default="static_dinov2_cls_patch_mean_failed_videos.txt",
    )
    parser.add_argument("--only_data_index", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--times_only", action="store_true")
    parser.add_argument("--finefs_stride", type=float, default=2.0)
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Optional HTTP/HTTPS proxy, e.g. http://127.0.0.1:7890",
    )
    args = parser.parse_args()

    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
        os.environ["ALL_PROXY"] = args.proxy

    if args.frames_per_second <= 0:
        raise ValueError("--frames_per_second must be positive")
    if args.sample_first_sec <= 0:
        raise ValueError("--sample_first_sec must be positive")
    if args.clip_len <= 0:
        raise ValueError("--clip_len must be positive")
    if args.infer_batch_size <= 0:
        raise ValueError("--infer_batch_size must be positive")
    if args.decode_batch_size <= 0:
        raise ValueError("--decode_batch_size must be positive")

    torch.backends.cudnn.benchmark = True
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    extractor = DinoV2StaticFeatureExtractor(
        device=device,
        model_name=args.model_name,
        output_dim=args.output_dim,
        image_size=args.image_size,
        infer_batch_size=args.infer_batch_size,
        decode_batch_size=args.decode_batch_size,
        use_amp=(not args.disable_amp),
        clip_len=args.clip_len,
        sample_first_sec=args.sample_first_sec,
        frames_per_second=args.frames_per_second,
        hub_dir=args.hub_dir,
    )

    cache_dir = os.path.join(args.root_path, args.cache_dir_name)
    failed_log_path = os.path.join(args.root_path, args.failed_log_name)

    if args.times_only:
        if args.dataset_mode != "fs1000":
            raise ValueError("--times_only currently applies to FS1000 caches")
        build_fs1000_time_sidecars(
            args.root_path,
            cache_dir,
            args.cache_prefix,
            args.clip_len,
            args.sample_first_sec,
        )
        return

    if args.dataset_mode == "finefs":
        precompute_finefs(
            root_path=args.root_path,
            extractor=extractor,
            cache_dir=cache_dir,
            cache_prefix=args.cache_prefix,
            failed_log_path=failed_log_path,
            only_data_index=args.only_data_index,
            limit=args.limit,
            overwrite=args.overwrite,
            seed=args.seed,
            stride=args.finefs_stride,
            save_frame_sequences=args.save_frame_sequences,
        )
        print("[action] DINOv2 FineFS precompute finished.", flush=True)
        return

    split_names = []
    if args.split in ("train", "all"):
        split_names.append("train")
    if args.split in ("val", "all"):
        split_names.append("val")

    for split_name in split_names:
        precompute_split(
            root_path=args.root_path,
            split_name=split_name,
            extractor=extractor,
            cache_dir=cache_dir,
            cache_prefix=args.cache_prefix,
            failed_log_path=failed_log_path,
            only_data_index=args.only_data_index,
            limit=args.limit,
            overwrite=args.overwrite,
            seed=args.seed,
            save_frame_sequences=args.save_frame_sequences,
        )

    print("[action] DINOv2 static feature precompute finished.", flush=True)


if __name__ == "__main__":
    main()
