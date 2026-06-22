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
        split_name = 'train_fs800.txt' if is_train else 'val_fs800.txt'
        file_path = os.path.join(root_path, split_name)
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

        audio_path = os.path.join(self.root_path, "new feature", "ast_feature_fs1000_new", data_index + '.npy')
        video_path = os.path.join(self.root_path, "Timesformer_output_feature_fs800", data_index + '.npy')

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
    def __init__(self, root_path, is_train=True, cache_dir_name="static_resnet50_cache", cache_prefix="static_resnet50", only_data_index=None):
        self.root_path = root_path
        self.cache_dir = os.path.join(root_path, cache_dir_name)
        self.cache_prefix = cache_prefix
        split_name = 'train_fs800.txt' if is_train else 'val_fs800.txt'
        file_path = os.path.join(root_path, split_name)
        with open(file_path, 'r') as f:
            data_info = f.readlines()

        self.total_data = []
        for data in data_info:
            data = data.strip('\n').split()
            if only_data_index is not None and data[0] != only_data_index:
                continue
            self.total_data.append(data)

    def __getitem__(self, index):
        data_info = self.total_data[index]
        data_index = data_info[0]

        audio_path = os.path.join(self.root_path, "new feature", "ast_feature_fs1000_new", data_index + '.npy')
        video_path = os.path.join(self.root_path, "Timesformer_output_feature_fs800", data_index + '.npy')
        audio_feature = torch.from_numpy(np.load(audio_path))
        video_feature = torch.from_numpy(np.load(video_path))

        t_dyn = min(audio_feature.shape[0], video_feature.shape[0])
        static_path = os.path.join(self.cache_dir, f"{self.cache_prefix}_{data_index}_T{t_dyn}.npy")
        if not os.path.exists(static_path):
            raise FileNotFoundError(f"Missing static feature cache: {static_path}")
        static_feature = torch.from_numpy(np.load(static_path))

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
