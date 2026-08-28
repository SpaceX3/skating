import torch
import torch.utils.data as data
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence

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
        else:
            file_path = root_path + 'val_fs800.txt'
        with open(file_path, 'r') as f:
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


class FeatureDatasetWithStaticCache(data.Dataset):
    def __init__(self, root_path, is_train=True, cache_dir_name="static_videomae_c1_first_token_cache", cache_prefix="static_videomae_c1_first_token", static_feature_dim=768):
        self.root_path = os.path.abspath(root_path)
        self.cache_dir = (
            cache_dir_name
            if os.path.isabs(cache_dir_name)
            else os.path.join(self.root_path, cache_dir_name)
        )
        self.cache_prefix = cache_prefix
        self.static_feature_dim = int(static_feature_dim)
        if is_train:
            file_path = os.path.join(self.root_path, 'train_fs800.txt')
        else:
            file_path = os.path.join(self.root_path, 'val_fs800.txt')
        with open(file_path, 'r') as f:
            data_info = f.readlines()

        self.total_data = []
        for data in data_info:
            data = data.strip('\n').split()
            self.total_data.append(data)

    def __getitem__(self, index):
        data_info = self.total_data[index]
        data_index = data_info[0]

        audio_path = os.path.join(
            self.root_path,
            "new feature",
            "ast_feature_fs1000_new",
            data_index + '.npy',
        )
        video_path = os.path.join(
            self.root_path,
            "Timesformer_output_feature_fs800",
            data_index + '.npy',
        )
        audio_feature = torch.from_numpy(np.load(audio_path))
        video_feature = torch.from_numpy(np.load(video_path))

        t_dyn = min(audio_feature.shape[0], video_feature.shape[0])
        static_path = os.path.join(self.cache_dir, f"{self.cache_prefix}_{data_index}_T{t_dyn}.npy")
        if not os.path.exists(static_path):
            raise FileNotFoundError(f"Missing static feature cache: {static_path}")
        static_values = np.load(static_path)
        expected_shape = (t_dyn, self.static_feature_dim)
        if static_values.shape != expected_shape:
            raise ValueError(
                f"static feature shape for {data_index} is {static_values.shape}, expected {expected_shape}"
            )
        static_feature = torch.from_numpy(static_values).float()

        tes = float(data_info[1])
        pcs = float(data_info[2])
        ss = float(data_info[3])
        trans = float(data_info[4])
        perform = float(data_info[5])
        composition = float(data_info[6])
        interpretation = float(data_info[7])
        factor = float(data_info[8])
        pcs = pcs / factor

        return audio_feature, video_feature, tes, pcs, ss, trans, perform, composition, interpretation, static_feature, data_index

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
    static_features = [item[9] for item in batch]
    data_index = [item[10] for item in batch]

    audio_len = [item[0].shape[0] for item in batch]
    video_len = [item[1].shape[0] for item in batch]

    audios = pad_sequence(audios, batch_first=True)
    audios = torch.unsqueeze(audios, dim=2)
    videos = pad_sequence(videos, batch_first=True)
    inv_audios = pad_sequence(inv_audios, batch_first=True)
    inv_audios = torch.unsqueeze(inv_audios, dim=2)
    inv_videos = pad_sequence(inv_videos, batch_first=True)
    static_features = pad_sequence(static_features, batch_first=True)

    tes = torch.FloatTensor(tes)
    pcs = torch.FloatTensor(pcs)
    ss = torch.FloatTensor(ss)
    trans = torch.FloatTensor(trans)
    perform = torch.FloatTensor(perform)
    composition = torch.FloatTensor(composition)
    interpretation = torch.FloatTensor(interpretation)
    scores = [tes, pcs, ss, trans, perform, composition, interpretation]

    return audios, videos, inv_audios, inv_videos, static_features, audio_len, video_len, scores, data_index


class FeatureDatasetWithVideoMAE(data.Dataset):
    def __init__(
        self,
        root_path,
        dynamic_cache_dir,
        is_train=True,
        dynamic_cache_prefix="dynamic_videomae_5x8",
        use_static_branch=False,
        static_cache_dir=None,
        static_cache_prefix="static_videomae_c1_top4_cross_attention",
        static_feature_dim=6914,
    ):
        self.root_path = os.path.abspath(root_path)
        self.dynamic_cache_dir = os.path.abspath(dynamic_cache_dir)
        self.dynamic_cache_prefix = dynamic_cache_prefix
        self.use_static_branch = bool(use_static_branch)
        self.static_cache_dir = os.path.abspath(static_cache_dir) if static_cache_dir else None
        self.static_cache_prefix = static_cache_prefix
        self.static_feature_dim = int(static_feature_dim)
        split_file = "train_fs800.txt" if is_train else "val_fs800.txt"
        with open(os.path.join(self.root_path, split_file), "r") as handle:
            self.total_data = [line.strip().split() for line in handle if line.strip()]

    def __len__(self):
        return len(self.total_data)

    def __getitem__(self, index):
        data_info = self.total_data[index]
        video_id = data_info[0]
        audio = torch.from_numpy(
            np.load(
                os.path.join(
                    self.root_path,
                    "new feature",
                    "ast_feature_fs1000_new",
                    video_id + ".npy",
                )
            )
        ).float()
        dynamic_length = len(audio)
        video_path = os.path.join(
            self.dynamic_cache_dir,
            f"{self.dynamic_cache_prefix}_{video_id}_T{dynamic_length}.npy",
        )
        video = torch.from_numpy(np.load(video_path)).float()
        if video.shape != (dynamic_length, 40, 768):
            raise ValueError(
                f"dynamic VideoMAE shape for {video_id} is {tuple(video.shape)}, "
                f"expected {(dynamic_length, 40, 768)}"
            )

        if self.use_static_branch:
            static_path = os.path.join(
                self.static_cache_dir,
                f"{self.static_cache_prefix}_{video_id}_T{dynamic_length}.npy",
            )
            static = torch.from_numpy(np.load(static_path)).float()
            if static.shape != (dynamic_length, self.static_feature_dim):
                raise ValueError(
                    f"static feature shape for {video_id} is {tuple(static.shape)}, "
                    f"expected {(dynamic_length, self.static_feature_dim)}"
                )
        else:
            static = torch.empty((dynamic_length, 0), dtype=torch.float32)

        values = [float(value) for value in data_info[1:9]]
        tes, pcs, ss, trans, perform, composition, interpretation, factor = values
        pcs /= factor
        return (
            audio,
            video,
            tes,
            pcs,
            ss,
            trans,
            perform,
            composition,
            interpretation,
            static,
            video_id,
        )

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
