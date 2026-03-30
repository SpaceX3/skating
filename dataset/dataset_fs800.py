import torch
import torch.utils.data as data
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random


def _safe_read_video_frame(cap, frame_idx):
    """
    Read a single frame at `frame_idx` from an opened cv2.VideoCapture.
    """
    import cv2  # lazy import: only needed if you use the static branch

    # Clamp to avoid out-of-range seeks
    if frame_idx < 0:
        frame_idx = 0
    # CAP_PROP_FRAME_COUNT might be unavailable in some containers; best-effort.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        return None
    return frame_bgr


class ResNet50StaticFeatureExtractor:
    """
    Extract per-frame static embeddings from raw video using ResNet50.
    Output: [T_dyn, 2048] float32
    """

    def __init__(
        self,
        device="cpu",
        cache_dir=None,
        cache_prefix="static_resnet50",
        static_in_dim=2048,
    ):
        self.device = device
        self.cache_dir = cache_dir
        self.cache_prefix = cache_prefix
        self.static_in_dim = static_in_dim

        self._resnet_features = None
        self._preprocess = None

    def _lazy_init(self):
        if self._resnet_features is not None:
            return

        import torch  # noqa: F401
        from torchvision import models
        from torchvision.models import ResNet50_Weights
        from torchvision import transforms
        from PIL import Image  # noqa: F401

        weights = ResNet50_Weights.IMAGENET1K_V2
        preprocess = weights.transforms()

        resnet = models.resnet50(weights=weights)
        # Remove classification head; keep global pooled embedding.
        resnet_features = torch.nn.Sequential(*list(resnet.children())[:-1])

        resnet_features.eval()
        self._resnet_features = resnet_features
        self._preprocess = preprocess

    def _get_cache_path(self, cache_key):
        if self.cache_dir is None:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{self.cache_prefix}_{cache_key}.npy")

    @torch.no_grad()
    def extract_from_video(
        self,
        video_path,
        T_dyn,
        is_train,
        cache_key,
        require_cache_only=False,
        deterministic_for_eval=True,
    ):
        """
        Sample exactly T_dyn frames by splitting the video into T_dyn bins.
        For each bin, pick 1 frame:
          - train: random frame in the bin
          - eval: center frame in the bin
        """
        import cv2  # lazy import: only needed if you use the static branch
        from PIL import Image

        cache_path = self._get_cache_path(cache_key)
        if cache_path is not None and os.path.exists(cache_path):
            feat = np.load(cache_path)
            if feat.shape[0] == T_dyn and feat.shape[1] == self.static_in_dim:
                return torch.from_numpy(feat)
        if require_cache_only:
            raise FileNotFoundError(
                f"Static cache missing for key={cache_key}. "
                "Please run precompute mode first to generate all static caches."
            )

        # Only initialize ResNet when we actually need to compute the cache.
        self._lazy_init()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Best-effort fallback: just sample first T_dyn frames.
            total_frames = T_dyn

        frame_indices = []
        for i in range(T_dyn):
            start = int(i * total_frames / T_dyn)
            end = int((i + 1) * total_frames / T_dyn) - 1
            if end < start:
                end = start
            start = max(0, min(start, total_frames - 1))
            end = max(0, min(end, total_frames - 1))

            if is_train:
                idx = random.randint(start, end)
            else:
                idx = (start + end) // 2
            frame_indices.append(idx)

        # Compute embeddings for unique indices to reduce repeated seeks.
        unique_indices = sorted(set(frame_indices))
        idx_to_feat = {}

        # Move model to device lazily (CPU by default).
        self._resnet_features = self._resnet_features.to(self.device)

        for idx in unique_indices:
            frame_bgr = _safe_read_video_frame(cap, idx)
            if frame_bgr is None:
                # Fallback: use zeros if decoding fails.
                idx_to_feat[idx] = np.zeros((self.static_in_dim,), dtype=np.float32)
                continue

            frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
            img = Image.fromarray(frame_rgb)
            x = self._preprocess(img).unsqueeze(0).to(self.device)
            emb = self._resnet_features(x)  # [1, 2048, 1, 1]
            emb = emb.flatten(1).detach().cpu().numpy().astype(np.float32)  # [1, 2048]
            idx_to_feat[idx] = emb[0]

        cap.release()

        feats = np.stack([idx_to_feat[idx] for idx in frame_indices], axis=0).astype(np.float32)

        if cache_path is not None:
            # Use a unique temp file to avoid multi-worker write collisions.
            tmp_path = cache_path + f".tmp.{os.getpid()}.{random.randint(0, 1_000_000)}.npy"
            np.save(tmp_path, feats)
            # Atomic move into place.
            os.replace(tmp_path, cache_path)

        return torch.from_numpy(feats)


class FeatureDatasetWithStaticResNet50(data.Dataset):
    """
    Like FeatureDataset, but also returns static ResNet50 per-frame embeddings
    sampled by splitting the raw video into T_dyn bins.
    """

    def __init__(
        self,
        root_path,
        is_train=True,
        device_for_static=None,
        cache_dir_name="static_resnet50_cache",
        require_cache_only=False,
    ):
        self.root_path = root_path
        self.is_train = is_train
        self.require_cache_only = require_cache_only

        if device_for_static is None:
            # Default to GPU if available, otherwise CPU.
            device_for_static = "cuda" if torch.cuda.is_available() else "cpu"

        file_path = os.path.join(root_path, "train_fs800.txt" if is_train else "val_fs800.txt")
        with open(file_path, "r") as f:
            data_info = f.readlines()

        self.total_data = []
        for data in data_info:
            data = data.strip("\n").split()
            self.total_data.append(data)

        cache_dir = os.path.join(root_path, cache_dir_name)
        self.extractor = ResNet50StaticFeatureExtractor(
            device=device_for_static,
            cache_dir=cache_dir,
            static_in_dim=2048,
        )

    def __getitem__(self, index):
        data_info = self.total_data[index]
        data_index = data_info[0]

        audio_path = os.path.join(
            self.root_path,
            "new feature",
            "ast_feature_fs1000_new",
            data_index + ".npy",
        )
        video_path = os.path.join(
            self.root_path,
            "Timesformer_output_feature_fs800",
            data_index + ".npy",
        )

        audio_feature = torch.from_numpy(np.load(audio_path))
        video_feature = torch.from_numpy(np.load(video_path))

        # Align static sampling with the dynamic time length.
        T_dyn = min(audio_feature.shape[0], video_feature.shape[0])

        mp4_path = os.path.join(self.root_path, "fs1000", f"{data_index}.mp4")
        cache_key = f"{data_index}_T{T_dyn}"
        static_feature = self.extractor.extract_from_video(
            mp4_path,
            T_dyn=T_dyn,
            is_train=self.is_train,
            cache_key=cache_key,
            require_cache_only=self.require_cache_only,
            deterministic_for_eval=True,
        )

        tes = float(data_info[1])
        pcs = float(data_info[2])
        ss = float(data_info[3])
        trans = float(data_info[4])
        perform = float(data_info[5])
        composition = float(data_info[6])
        interpretation = float(data_info[7])
        factor = float(data_info[8])

        pcs = pcs / factor

        return (
            audio_feature,
            video_feature,
            tes,
            pcs,
            ss,
            trans,
            perform,
            composition,
            interpretation,
            static_feature,  # [T_dyn, 2048]
            data_index,
        )

    def __len__(self):
        return len(self.total_data)


def av_collate_fn_with_static(batch):
    audios = [item[0] for item in batch]
    videos = [item[1] for item in batch]
    inv_audios = [torch.flip(item[0], [0]) for item in batch]
    inv_videos = [torch.flip(item[1], [0]) for item in batch]

    tes = [item[2] for item in batch]
    pcs = [item[3] for item in batch]
    ss = [item[4] for item in batch]
    trans = [item[5] for item in batch]
    perform = [item[6] for item in batch]
    composition = [item[7] for item in batch]
    interpretation = [item[8] for item in batch]

    static_features = [item[9] for item in batch]  # [T_dyn, 2048]
    data_index = [item[10] for item in batch]

    audio_len = [item[0].shape[0] for item in batch]
    video_len = [item[1].shape[0] for item in batch]

    audios = pad_sequence(audios, batch_first=True)  # [B, T, 768]
    audios = torch.unsqueeze(audios, dim=2)  # [B, T, 1, 768]

    videos = pad_sequence(videos, batch_first=True)  # [B, T, 15, 768]

    inv_audios = pad_sequence(inv_audios, batch_first=True)
    inv_audios = torch.unsqueeze(inv_audios, dim=2)

    inv_videos = pad_sequence(inv_videos, batch_first=True)

    static_features = pad_sequence(static_features, batch_first=True)  # [B, T, 2048]

    tes = torch.FloatTensor(tes)
    pcs = torch.FloatTensor(pcs)
    ss = torch.FloatTensor(ss)
    trans = torch.FloatTensor(trans)
    perform = torch.FloatTensor(perform)
    composition = torch.FloatTensor(composition)
    interpretation = torch.FloatTensor(interpretation)

    scores = [tes, pcs, ss, trans, perform, composition, interpretation]
    return audios, videos, inv_audios, inv_videos, static_features, audio_len, video_len, scores, data_index


class FeatureDatasetWithStaticCache(data.Dataset):
    """
    Training/inference dataset that ONLY loads precomputed static features from disk.
    No video decoding, no ResNet forward here.
    """

    def __init__(
        self,
        root_path,
        is_train=True,
        static_cache_dir_name="static_resnet50_cache",
        static_cache_prefix="static_resnet50",
        strict_cache=True,
    ):
        self.root_path = root_path
        self.is_train = is_train
        self.static_cache_prefix = static_cache_prefix
        self.strict_cache = strict_cache
        self.static_cache_dir = os.path.join(root_path, static_cache_dir_name)

        file_path = os.path.join(root_path, "train_fs800.txt" if is_train else "val_fs800.txt")
        with open(file_path, "r") as f:
            data_info = f.readlines()

        self.total_data = []
        for data in data_info:
            self.total_data.append(data.strip("\n").split())

    def _cache_path(self, data_index, T_dyn):
        cache_key = f"{data_index}_T{T_dyn}"
        return os.path.join(self.static_cache_dir, f"{self.static_cache_prefix}_{cache_key}.npy")

    def __getitem__(self, index):
        data_info = self.total_data[index]
        data_index = data_info[0]

        audio_path = os.path.join(self.root_path, "new feature", "ast_feature_fs1000_new", data_index + ".npy")
        video_path = os.path.join(self.root_path, "Timesformer_output_feature_fs800", data_index + ".npy")

        audio_feature = torch.from_numpy(np.load(audio_path))
        video_feature = torch.from_numpy(np.load(video_path))

        T_dyn = min(audio_feature.shape[0], video_feature.shape[0])
        static_path = self._cache_path(data_index, T_dyn)

        if (not os.path.exists(static_path)) and self.strict_cache:
            raise FileNotFoundError(
                f"Static cache not found: {static_path}. "
                "Please run action.py first to precompute static features."
            )
        if os.path.exists(static_path):
            static_feature = torch.from_numpy(np.load(static_path))
        else:
            static_feature = torch.zeros((T_dyn, 2048), dtype=torch.float32)

        tes = float(data_info[1])
        pcs = float(data_info[2])
        ss = float(data_info[3])
        trans = float(data_info[4])
        perform = float(data_info[5])
        composition = float(data_info[6])
        interpretation = float(data_info[7])
        factor = float(data_info[8])
        pcs = pcs / factor

        return (
            audio_feature,
            video_feature,
            tes,
            pcs,
            ss,
            trans,
            perform,
            composition,
            interpretation,
            static_feature,
            data_index,
        )

    def __len__(self):
        return len(self.total_data)

class FeatureDataset(data.Dataset):
    def __init__(self,
                 root_path,
                 is_train=True,
                 ):

        # self.spatial_transform = spatial_transform
        # self.temporal_transform = temporal_transform
        self.root_path = root_path
        if is_train:
            file_path = root_path + 'train_fs800.txt'
            f = open(file_path, 'r')
            data_info = f.readlines()
        else:
            file_path = root_path + 'val_fs800.txt'
            f = open(file_path, 'r')
            data_info = f.readlines()

        self.total_data = []
        for data in data_info:
            data = data.strip('\n')
            data = data.split()
            self.total_data.append(data)
            
        # print(self.total_data)
        
    
    def __getitem__(self, index):
        data_info = self.total_data[index]
        data_index = data_info[0]
        
        audio_path = "../FS1000 Dataset/new feature/ast_feature_fs1000_new/" + data_index + '.npy'
        video_path = "../FS1000 Dataset/Timesformer_output_feature_fs800/" + data_index + '.npy'

        audio_feature = torch.from_numpy(np.load(audio_path))
        video_feature = torch.from_numpy(np.load(video_path))
        
        tes = float(data_info[1])
        pcs = float(data_info[2])

        ss = float(data_info[3])
        trans = float(data_info[4])
        perform = float(data_info[5])
        composition = float(data_info[6])
        interpretation = float(data_info[7])
        factor = float(data_info[8])

        pcs = pcs / factor


        return audio_feature, video_feature, tes, pcs, ss, trans, perform, composition, interpretation, factor, data_index
        # return tes, pcs

    def __len__(self):
        return len(self.total_data)

    
def av_collate_fn(batch):
    audios = [item[0] for item in batch]
    videos = [item[1] for item in batch]
    inv_audios = [torch.flip(item[0], [0]) for item in batch]
    inv_videos = [torch.flip(item[1], [0]) for item in batch]
    tes = [item[2] for item in batch]
    pcs = [item[3] for item in batch]
    ss = [item[4] for item in batch]
    trans = [item[5] for item in batch]
    perform = [item[6] for item in batch]
    composition = [item[7] for item in batch]
    interpretation = [item[8] for item in batch]
    # factor = [item[9] for item in batch]
    data_index = [item[10] for item in batch]
    
    audio_len = [item[0].shape[0] for item in batch]
    video_len = [item[1].shape[0] for item in batch]

    # for i in range(len(audios)):
    #     print(audios[i].shape)
    #     print(videos[i].shape)
    
    # exit()
    # print(audios[0].shape)
    # print(audios[0][0])
    # print(audios[0][0].shape)
    # print(audios[0][-1]==inv_audios[0][0])
    # print(inv_audios[0])
    # exit()
    audios = pad_sequence(audios, batch_first=True)
    # audios = torch.nn.functional.pad(audios, (0, 0, 0, 163-audios.shape[1]), 'constant', 0)
    audios = torch.unsqueeze(audios, dim=2)
    videos = pad_sequence(videos, batch_first=True)

    inv_audios = pad_sequence(inv_audios, batch_first=True)
    inv_audios = torch.unsqueeze(inv_audios, dim=2)
    inv_videos = pad_sequence(inv_videos, batch_first=True)
    # videos = torch.nn.functional.pad(videos, (0, 0, 0, 1190-videos.shape[1]), 'constant', 0)

    tes = torch.FloatTensor(tes)
    pcs = torch.FloatTensor(pcs)
    ss = torch.FloatTensor(ss)
    trans = torch.FloatTensor(trans)
    perform = torch.FloatTensor(perform)
    composition = torch.FloatTensor(composition)
    interpretation = torch.FloatTensor(interpretation)
    # factor = torch.FloatTensor(factor)

    # pcs_unfac = pcs / factor

    scores = [tes, pcs, ss, trans, perform, composition, interpretation]

    return audios, videos, inv_audios, inv_videos, audio_len, video_len, scores, data_index



if __name__=='__main__':
    dataset = FeatureDataset("/data1/xiajingfei/data", is_train=False)
    dataloader = data.DataLoader(dataset, batch_size=50, collate_fn=av_collate_fn)
    # a = np.load('/data1/xiajingfei/project/ast/output_feature_fs800/2019_SWJ_LF_Alexandra_T.npy')
    # print(a.shape)
    count = 0
    for a, b, m, n, c, d, e, f, g in dataloader:
        # print(a.shape)
        # print(b.shape)
        # print(c)
        # print(d)
        # print(e)
        # print(f)
        # print(g)
        # print(max(d))
        # print(max(c))
        for i in range(len(c)):
            if c[i] != d[i]:
                # if c[i] - 1 == d[i]:
                count += 1
                print("al: ", c[i], "vl: ", d[i], "idx: ", g[i])
        # break
    print(count)
    # a = torch.randn(3, 4, 5)
    # x = torch.flip(a, [0])
    # print(a[1])
    # print(x[1])