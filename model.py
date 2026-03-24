# from librosa.core import audio
# from numpy.ma import clip
from cv2 import transform
from torch import nn
from functools import partial
# from einops.layers.torch import Rearrange, Reduce
import torch
import time
import math


def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn, transpose=False):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.transpose = transpose

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class scoring_head(nn.Module):
    def __init__(
        self,
        depth,
        input_dim,
        dim,
        input_len=2,
        num_scores=1,
        memory_size=64,
        write_stride=4,
        use_global_memory=True,
        global_memory_slots=32,
        global_momentum=0.99,
        freeze_global_on_eval=True,
    ):
        super().__init__()


        self.init_memory_key = nn.parameter.Parameter(torch.randn(1, dim))
        self.init_memory_value = nn.parameter.Parameter(torch.randn(1, dim))
        self.cls_token = nn.parameter.Parameter(torch.randn(1, dim))
        self.memory_size = memory_size
        self.write_stride = max(int(write_stride), 1)
        self.use_global_memory = use_global_memory
        self.global_memory_slots = max(int(global_memory_slots), 1)
        self.global_momentum = float(global_momentum)
        self.freeze_global_on_eval = freeze_global_on_eval

        self.linear1 = nn.Linear(input_dim, dim)
        self.mem_query = nn.Linear(dim, dim)
        self.mem_key = nn.Linear(dim, dim)
        self.mem_value = nn.Linear(dim, dim)
        self.global_mem_key_proj = nn.Linear(dim, dim)
        self.global_mem_value_proj = nn.Linear(dim, dim)
        
        self.linear_forward = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward((input_len + 2), dense = partial(nn.Conv1d, kernel_size=1))),  # token mixing
                PreNormResidual(dim, FeedForward(dim))) for _ in range(depth)]    # channel mixing
        )

        self.layer_norm = nn.LayerNorm(dim)

        self.hidden_linear = nn.Linear(dim, dim)
        self.output = head(dim, num_scores)

        self.register_buffer("global_memory_key", torch.randn(self.global_memory_slots, dim))
        self.register_buffer("global_memory_value", torch.randn(self.global_memory_slots, dim))
        self.register_buffer("global_memory_ptr", torch.zeros(1, dtype=torch.long))

        # time-wise scoring head: 2 hidden layers MLP applied on each timestep feature
        # input: [B, T, dim] -> output: [B, T, num_scores]
        hidden1 = dim
        hidden2 = max(dim // 2, 1)
        self.time_score_mlp = nn.Sequential(
            nn.Linear(dim, hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Linear(hidden2, num_scores),
        )

        # for c3d & vggish
        # self.video_transform_linear = nn.Linear(input_dim, dim)
        # self.audio_transform_linear = nn.Linear(input_dim, dim)

    def _init_memory_bank(self, batch_size, device, dtype):
        memory_key = self.init_memory_key.unsqueeze(dim=0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        memory_value = self.init_memory_value.unsqueeze(dim=0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        return memory_key, memory_value

    def _read_memory(self, query, memory_key, memory_value):
        # query: [B, 1, D], memory_key/value: [B, M, D]
        attn_score = torch.matmul(query, memory_key.transpose(1, 2)) / math.sqrt(query.size(-1))
        attn_weight = torch.softmax(attn_score, dim=-1)
        memory_context = torch.matmul(attn_weight, memory_value)
        return memory_context

    def _update_memory_bank(self, memory_key, memory_value, new_key, new_value):
        memory_key = torch.cat([memory_key, new_key], dim=1)
        memory_value = torch.cat([memory_value, new_value], dim=1)
        if memory_key.size(1) > self.memory_size:
            memory_key = memory_key[:, -self.memory_size:]
            memory_value = memory_value[:, -self.memory_size:]
        return memory_key, memory_value

    def _read_global_memory(self, query):
        if not self.use_global_memory:
            return torch.zeros_like(query)

        # Use cloned snapshots so in-forward EMA updates on global buffers do not
        # trigger autograd version mismatch for tensors saved for backward.
        key_snapshot = self.global_memory_key.detach().clone()
        value_snapshot = self.global_memory_value.detach().clone()
        global_key = self.global_mem_key_proj(key_snapshot).to(device=query.device, dtype=query.dtype)
        global_value = self.global_mem_value_proj(value_snapshot).to(device=query.device, dtype=query.dtype)
        global_key = global_key.unsqueeze(0).expand(query.size(0), -1, -1)
        global_value = global_value.unsqueeze(0).expand(query.size(0), -1, -1)
        return self._read_memory(query, global_key, global_value)

    def _update_global_memory(self, new_key, new_value):
        if not self.use_global_memory:
            return
        if self.freeze_global_on_eval and (not self.training):
            return

        with torch.no_grad():
            key_avg = new_key.detach().mean(dim=(0, 1))
            value_avg = new_value.detach().mean(dim=(0, 1))
            slot = int(self.global_memory_ptr.item() % self.global_memory_slots)
            self.global_memory_key[slot] = self.global_momentum * self.global_memory_key[slot] + (1.0 - self.global_momentum) * key_avg
            self.global_memory_value[slot] = self.global_momentum * self.global_memory_value[slot] + (1.0 - self.global_momentum) * value_avg
            self.global_memory_ptr[0] = (self.global_memory_ptr[0] + 1) % self.global_memory_slots

    def forward(self, audio_feature, video_feature, inv_audio_feature, inv_video_feature, audio_len, video_len):
        batch_size, aclip, _, _ = audio_feature.shape
        batch_size, vclip, _, _ = video_feature.shape
        clip = min(aclip, vclip)

        hidden_states = []
        back_hidden_states = []
        fwd_memory_key, fwd_memory_value = self._init_memory_bank(batch_size, audio_feature.device, audio_feature.dtype)
        bwd_memory_key, bwd_memory_value = self._init_memory_bank(batch_size, audio_feature.device, audio_feature.dtype)

        for j in range(clip):
            
            curr_audio_feature = audio_feature[:, j]
            curr_video_feature = video_feature[:, j]
            # curr_audio_feature = self.audio_transform_linear(curr_audio_feature)
            # curr_video_feature = self.video_transform_linear(curr_video_feature)
            input_feature = torch.cat([curr_audio_feature, curr_video_feature], dim=1)

            back_curr_audio_feature = inv_audio_feature[:, j]
            back_curr_video_feature = inv_video_feature[:, j]
            # back_curr_audio_feature = self.audio_transform_linear(back_curr_audio_feature)
            # back_curr_video_feature = self.video_transform_linear(back_curr_video_feature)
            back_input_feature = torch.cat([back_curr_audio_feature, back_curr_video_feature], dim=1)
            
            cls, new_fwd_key, new_fwd_value = self.model_forward(input_feature, fwd_memory_key, fwd_memory_value)
            back_cls, new_bwd_key, new_bwd_value = self.model_forward(back_input_feature, bwd_memory_key, bwd_memory_value, back=True)

            if j % self.write_stride == 0:
                fwd_memory_key, fwd_memory_value = self._update_memory_bank(fwd_memory_key, fwd_memory_value, new_fwd_key, new_fwd_value)
                bwd_memory_key, bwd_memory_value = self._update_memory_bank(bwd_memory_key, bwd_memory_value, new_bwd_key, new_bwd_value)
                self._update_global_memory(torch.cat([new_fwd_key, new_bwd_key], dim=0), torch.cat([new_fwd_value, new_bwd_value], dim=0))

            hidden_states.append(cls)
            back_hidden_states.insert(0, back_cls)
                

        final_scores = []
        for i in range(batch_size):
            curr_batch_audio_len = audio_len[i]
            curr_batch_video_len = video_len[i]
            curr_batch_len = min(curr_batch_audio_len, curr_batch_video_len)

            # [1, T, dim]
            cl = torch.cat(hidden_states[:curr_batch_len], dim=1)[i:i+1]
            bk_cl = torch.cat(back_hidden_states[:curr_batch_len], dim=1)[i:i+1]

            # do NOT average over time; average cl/bk_cl at each timestep
            time_feat = (cl + bk_cl) / 2  # [1, T, dim]

            # MLP predicts score per timestep, then aggregate to scalar score
            time_score = self.time_score_mlp(time_feat).squeeze(-1)  # [1, T]
            batch_score = torch.mean(time_score, dim=1)  # [1]
            final_scores.append(batch_score)

        output = torch.cat(final_scores, dim=0)  # [B]
        return output
    
    def model_forward(self, x, memory_key, memory_value, back=False):
        # x shape: B x 2 (a & v) x D

        x = self.linear1(x)
        query = self.mem_query(torch.mean(x, dim=1, keepdim=True))
        memory_context = self._read_memory(query, memory_key, memory_value)
        global_context = self._read_global_memory(query)
        memory_context = (memory_context + global_context) / 2

        if back:
            batch_size = x.shape[0]
            back_cls_token = self.cls_token.unsqueeze(dim=0)
            cls_token = torch.cat([back_cls_token for _ in range(batch_size)], dim=0)

            concat_input = torch.cat([cls_token, x, memory_context], dim=1)

        else:
            batch_size = x.shape[0]
            cls_token = self.cls_token.unsqueeze(dim=0)
            cls_token = torch.cat([cls_token for _ in range(batch_size)], dim=0)

            concat_input = torch.cat([memory_context, x, cls_token], dim=1)
        
        out = self.linear_forward(concat_input)

        if back:
            out_cls = out[:, 0:1]
        else:
            out_cls = out[:, -1:]

        out_cls = self.hidden_linear(out_cls)
        new_memory_key = self.mem_key(query)
        new_memory_value = self.mem_value(out_cls)

        return out_cls, new_memory_key, new_memory_value


class head(nn.Module):
    def __init__(self, dim, num_scores=1):
        super().__init__()
        self.linear = nn.Linear(dim, num_scores)

    def forward(self, x):
        x = self.linear(x)
        return x
