import os
import argparse
import numpy as np
import torch
from torch import nn


def try_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None


def load_ast_backbone(device: str = "cuda:0", ckpt_dir: str = None):
    """
    优先从本地目录加载 AST（Audio Spectrogram Transformer）backbone。
    - ckpt_dir: 包含 config.json / pytorch_model.bin 等（如 ./model/ast-base-audioset）
    返回: dict 或 None（加载失败时）
    """
    # 优先尝试 ONNX Runtime 模型（目录内存在 .onnx）
    onnxruntime = try_import("onnxruntime")
    if ckpt_dir is not None and os.path.isdir(ckpt_dir):
        onnx_paths = []
        for root, _, files in os.walk(ckpt_dir):
            for f in files:
                if f.endswith(".onnx"):
                    onnx_paths.append(os.path.join(root, f))
        if onnx_paths and onnxruntime is not None:
            # 优先非 fp16 模型，尽量避免某些 ORT 图优化兼容问题
            onnx_paths = sorted(onnx_paths, key=lambda p: ("fp16" in os.path.basename(p).lower(), p))
            for onnx_path in onnx_paths:
                print(f"[audio_preproc] 检测到 ONNX AST 模型: {onnx_path}")
                # 先尝试 CUDA+CPU，再尝试纯 CPU；两者都失败则换下一个 onnx 文件
                provider_candidates = [
                    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    ["CPUExecutionProvider"],
                ]
                for providers in provider_candidates:
                    try:
                        _sess = onnxruntime.InferenceSession(onnx_path, providers=providers)
                        inputs = _sess.get_inputs()
                        outputs = _sess.get_outputs()
                        input_name = inputs[0].name
                        input_shape = inputs[0].shape  # e.g., [None, 160000] or dynamic
                        output_name = outputs[0].name
                        print(f"[audio_preproc] ONNX AST loaded with providers={providers}")
                        return {
                            "type": "onnx",
                            "session": _sess,
                            "providers": providers,
                            "input_name": input_name,
                            "input_shape": input_shape,
                            "output_name": output_name,
                            "target_sr": 16000,
                        }
                    except Exception as e:
                        print(f"[audio_preproc][warn] ONNX session init failed ({providers}): {e}")
                        continue
            print("[audio_preproc][warn] 所有 ONNX AST 模型均初始化失败，将尝试 HF AST / log-mel 回退。")

    transformers = try_import("transformers")
    if transformers is None:
        print("[audio_preproc][warn] transformers 未安装，AST 将不可用，退回到 log-mel 特征。")
        return None

    from transformers import AutoFeatureExtractor, ASTModel
    try:
        if ckpt_dir is None:
            # 未提供本地权重目录时，不尝试联网下载；直接返回 None
            print("[audio_preproc][warn] 未提供 ast_ckpt_dir，本地 AST 权重不可用，退回到 log-mel 特征。")
            return None
        # 如果给的是上层目录（如 ./model/ast），尝试在其子目录中寻找包含 config.json 的权重目录
        search_dir = ckpt_dir
        if os.path.isdir(ckpt_dir) and not os.path.isfile(os.path.join(ckpt_dir, "config.json")):
            candidates = []
            for name in os.listdir(ckpt_dir):
                cand = os.path.join(ckpt_dir, name)
                if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "config.json")):
                    candidates.append(cand)
            if candidates:
                # 选择第一个匹配的
                search_dir = candidates[0]
                print(f"[audio_preproc] 使用检测到的 AST 权重目录: {search_dir}")
            else:
                print(f"[audio_preproc][warn] 在 {ckpt_dir} 下未找到包含 config.json 的子目录，退回到 log-mel 特征。")
                return None

        feature_extractor = AutoFeatureExtractor.from_pretrained(search_dir, local_files_only=True)
        model = ASTModel.from_pretrained(search_dir, local_files_only=True)
        model.eval().to(device)
        hidden_size = int(getattr(model.config, "hidden_size", 768))
        return {
            "type": "hf",
            "feature_extractor": feature_extractor,
            "model": model,
            "hidden_size": hidden_size,
        }
    except Exception as e:
        print(f"[audio_preproc][warn] 本地 AST 权重加载失败，退回到 log-mel 特征: {e}")
        return None


def load_audio_from_mp4(video_path: str, target_sr: int = 16000):
    """
    从 .mp4 文件中读取音频波形（单声道，target_sr）。
    优先 librosa -> torchaudio；均不可用则返回 None
    """
    librosa = try_import("librosa")
    if librosa is not None:
        try:
            y, sr = librosa.load(video_path, sr=target_sr, mono=True)
            return y, target_sr
        except Exception:
            pass
    torchaudio = try_import("torchaudio")
    if torchaudio is not None:
        try:
            # 使用 StreamReader 读取音频流
            from torchaudio.io import StreamReader
            s = StreamReader(src=video_path)
            s.add_audio_stream(frames_per_chunk=0, sample_rate=target_sr)
            chunks = []
            for (audio_chunk,) in s.stream():
                # audio_chunk: [num_frames, num_channels]
                if audio_chunk.dim() == 2 and audio_chunk.shape[1] > 1:
                    audio_chunk = torch.mean(audio_chunk, dim=1, keepdim=True)
                chunks.append(audio_chunk.squeeze(-1).cpu())
            if chunks:
                y = torch.cat(chunks, dim=0).numpy()
                return y, target_sr
        except Exception:
            pass
    print(f"[audio_preproc][warn] 无法从 mp4 读取音频，跳过: {video_path}")
    return None, None


def segment_audio(y: np.ndarray, sr: int, clip_len_sec: float, overlap_ratio: float):
    """
    将整段音频切分为多个 clip（与 preproc 相同 overlap 策略）
    返回: List[np.ndarray]，每段长度近似 clip_len_sec
    """
    n_total = len(y)
    clip_len = int(clip_len_sec * sr)
    stride = int(clip_len_sec * (1.0 - overlap_ratio) * sr)
    if stride <= 0:
        stride = clip_len
    segments = []
    start = 0
    while start < n_total:
        end = min(start + clip_len, n_total)
        seg = y[start:end]
        if len(seg) > 0:
            segments.append(seg)
        if end == n_total:
            break
        start += stride
    return segments


def ast_feature_from_chunks(ast, chunks, sr, device: str, batch_size: int):
    """
    使用 AST 对音频 chunks 提取特征；返回 [T_clip, D]
    - ast: dict from load_ast_backbone
    """
    import torch
    if ast.get("type") == "onnx":
        sess = ast["session"]
        input_name = ast["input_name"]
        input_shape = ast["input_shape"]
        output_name = ast["output_name"]
        # 解析目标长度（如果静态 shape）
        # 典型输入为 [B, L] 或 [B, 1, L]；若为 dynamic，则用 clip 长度自适应（统一 pad 到批内最大）
        feats = []
        idx = 0
        while idx < len(chunks):
            batch = chunks[idx:idx + batch_size]
            # 统一到批内最大长度，或静态指定长度
            if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2 and isinstance(input_shape[-1], int):
                L = input_shape[-1]
            else:
                L = max(len(x) for x in batch)
            batch_arr = []
            for y in batch:
                if len(y) < L:
                    pad = np.zeros((L - len(y),), dtype=np.float32)
                    arr = np.concatenate([y.astype(np.float32), pad], axis=0)
                else:
                    arr = y[:L].astype(np.float32)
                batch_arr.append(arr[None, :])
            inp = np.concatenate(batch_arr, axis=0)  # [B,L]
            # 若模型期望 [B,1,L]
            need_3d = False
            if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
                need_3d = True
            if need_3d:
                inp = inp[:, None, :]  # [B,1,L]
            out = sess.run([output_name], {input_name: inp})[0]  # [B,D] or [B,T,D]
            out = np.asarray(out)
            if out.ndim == 3:
                out = out.mean(axis=1)  # [B,D]
            feats.append(out.astype(np.float32))
            idx += batch_size
        if not feats:
            return None
        return np.concatenate(feats, axis=0)  # [T_clip, D]
    else:
        feature_extractor = ast["feature_extractor"]
        model = ast["model"]
        feats = []
        with torch.no_grad():
            for s in range(0, len(chunks), batch_size):
                batch = chunks[s:s + batch_size]
                # feature extractor 接受 List[np.ndarray]
                inputs = feature_extractor(batch, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = inputs["input_values"].to(device)
                outputs = model(input_values=input_values)
                # 使用 pooled 或 CLS/mean
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    emb = outputs.pooler_output  # [B, D]
                else:
                    last_hidden = outputs.last_hidden_state  # [B, T, D]
                    emb = torch.mean(last_hidden, dim=1)  # [B, D]
                feats.append(emb.detach().cpu())
        if not feats:
            return None
        feats = torch.cat(feats, dim=0).numpy().astype(np.float32)  # [T_clip, D]
        return feats


def logmel_feature_from_chunks(chunks, sr, target_dim: int):
    """
    退而求其次：用 log-mel 作为音频特征，并映射/裁剪到 target_dim 维。
    返回 [T_clip, target_dim]
    """
    librosa = try_import("librosa")
    if librosa is None:
        print("[audio_preproc][error] librosa 未安装，无法生成备选 log-mel 特征。")
        return None
    out = []
    for y in chunks:
        # 生成 log-mel（长度因 clip 时长而异），取时频均值 -> 向量
        try:
            melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr // 2)
            logmel = librosa.power_to_db(melspec, ref=np.max)  # [128, Tm]
            vec = np.mean(logmel, axis=1)  # [128]
            # pad/裁剪到 target_dim
            if vec.shape[0] < target_dim:
                tmp = np.zeros((target_dim,), dtype=np.float32)
                tmp[:vec.shape[0]] = vec.astype(np.float32)
                vec = tmp
            elif vec.shape[0] > target_dim:
                vec = vec[:target_dim].astype(np.float32)
            else:
                vec = vec.astype(np.float32)
            out.append(vec[None, :])
        except Exception:
            out.append(np.zeros((1, target_dim), dtype=np.float32))
    if not out:
        return None
    return np.concatenate(out, axis=0)  # [T_clip, target_dim]


def process_one_video_ast(
    video_path: str,
    out_path: str,
    clip_len_sec: float,
    overlap_ratio: float,
    device: str,
    ast_cfg: dict,
    target_dim: int,
    batch_size: int,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    y, sr = load_audio_from_mp4(video_path, target_sr=16000)
    if y is None:
        return
    chunks = segment_audio(y, sr, clip_len_sec=clip_len_sec, overlap_ratio=overlap_ratio)
    if not chunks:
        return
    if ast_cfg is not None:
        feats = ast_feature_from_chunks(ast_cfg, chunks, sr, device=device, batch_size=batch_size)
    else:
        feats = logmel_feature_from_chunks(chunks, sr, target_dim=target_dim)
    if feats is None:
        return
    # 保存为 [T_clip, target_dim]
    np.save(out_path, feats.astype(np.float32))
    print(f"[audio_preproc] saved feature: {out_path}, shape={feats.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../FS1000 Dataset/")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "both"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--clip_len", type=float, default=5.0)
    parser.add_argument("--overlap_ratio", type=float, default=0.5)
    parser.add_argument("--ast_ckpt_dir", type=str, default="./model/ast-finetuned-audioset-10-10-0.4593", help="本地 AST 权重目录或其父目录（含 config.json/pytorch_model.bin）")
    parser.add_argument("--infer_batch_size", type=int, default=32)
    parser.add_argument("--target_dim", type=int, default=768, help="与现有 audio_feature 的通道维一致")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    ast_cfg = load_ast_backbone(device=device, ckpt_dir=args.ast_ckpt_dir)
    target_dim = int(args.target_dim if ast_cfg is None else getattr(ast_cfg["model"].config, "hidden_size", args.target_dim))

    def process_split(split_name: str):
        txt_path = os.path.join(args.root, f"{split_name}_fs800.txt")
        out_dir = os.path.join(args.root, "AST_feature_1000")
        os.makedirs(out_dir, exist_ok=True)
        with open(txt_path, "r") as f:
            lines = [line.strip().split() for line in f.readlines()]
        for i, data in enumerate(lines):
            data_index = data[0]
            video_path = os.path.join(args.root, "fs1000", f"{data_index}.mp4")
            out_path = os.path.join(out_dir, f"{data_index}.npy")
            if os.path.exists(out_path):
                continue
            process_one_video_ast(
                video_path=video_path,
                out_path=out_path,
                clip_len_sec=args.clip_len,
                overlap_ratio=args.overlap_ratio,
                device=device,
                ast_cfg=ast_cfg,
                target_dim=target_dim,
                batch_size=args.infer_batch_size,
            )
            if (i + 1) % 20 == 0:
                print(f"[audio_preproc] {split_name}: {i+1}/{len(lines)} processed")

    if args.split in ("train", "both"):
        process_split("train")
    if args.split in ("val", "both"):
        process_split("val")


if __name__ == "__main__":
    main()

