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
        use_static_branch=False,
        static_in_dim=2048,
        memory_size=64,
        write_stride=4,
        use_rag_memory=True,
        rag_memory_capacity=8192,
        rag_topk=16,
        rag_update_train_only=True,
        rag_memory_dropout=0.1,
    ):
        super().__init__()
        self.use_static_branch = use_static_branch
        self.static_in_dim = static_in_dim


        self.init_memory_key = nn.parameter.Parameter(torch.randn(1, dim))
        self.init_memory_value = nn.parameter.Parameter(torch.randn(1, dim))
        self.cls_token = nn.parameter.Parameter(torch.randn(1, dim))
        self.memory_size = memory_size
        self.write_stride = max(int(write_stride), 1)
        self.use_rag_memory = use_rag_memory
        self.rag_memory_capacity = max(int(rag_memory_capacity), 1)
        self.rag_topk = max(int(rag_topk), 1)
        self.rag_update_train_only = rag_update_train_only
        self.rag_memory_dropout = float(rag_memory_dropout)

        self.linear1 = nn.Linear(input_dim, dim)
        self.mem_query = nn.Linear(dim, dim)
        self.mem_key = nn.Linear(dim, dim)
        self.mem_value = nn.Linear(dim, dim)
        self.rag_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        
        self.linear_forward = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward((input_len + 2), dense = partial(nn.Conv1d, kernel_size=1))),  # token mixing
                PreNormResidual(dim, FeedForward(dim))) for _ in range(depth)]    # channel mixing
        )

        self.layer_norm = nn.LayerNorm(dim)

        self.hidden_linear = nn.Linear(dim, dim)
        self.output = head(dim, num_scores)

        self.register_buffer("rag_memory_key", torch.randn(self.rag_memory_capacity, dim))
        self.register_buffer("rag_memory_value", torch.randn(self.rag_memory_capacity, dim))
        self.register_buffer("rag_memory_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("rag_memory_ptr", torch.zeros(1, dtype=torch.long))

        # time-wise scoring head: applied on each timestep fused feature
        # input: [B, T, dim] -> output: [B, T, num_scores]
        hidden1 = dim
        hidden2 = max(dim // 2, 1)

        fused_dim = dim * 2 if self.use_static_branch else dim
        if self.use_static_branch:
            # Project static embedding to the same dim as dynamic embedding before concat.
            self.static_proj = nn.Linear(static_in_dim, dim)

        self.time_score_mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden1),
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

    def _read_rag_memory(self, query):
        if not self.use_rag_memory:
            return torch.zeros_like(query)

        count = int(self.rag_memory_count.item())
        if count <= 0:
            return torch.zeros_like(query)

        key_snapshot = self.rag_memory_key[:count].detach().clone().to(device=query.device, dtype=query.dtype)
        value_snapshot = self.rag_memory_value[:count].detach().clone().to(device=query.device, dtype=query.dtype)

        score = torch.matmul(query, key_snapshot.t().unsqueeze(0)) / math.sqrt(query.size(-1))
        topk = min(self.rag_topk, count)
        top_score, top_idx = torch.topk(score, k=topk, dim=-1)
        attn = torch.softmax(top_score, dim=-1)
        selected_value = value_snapshot[top_idx.squeeze(1)]
        context = torch.sum(selected_value * attn.transpose(1, 2), dim=1, keepdim=True)
        return context

    def _update_rag_memory(self, new_key, new_value):
        if not self.use_rag_memory:
            return
        if self.rag_update_train_only and (not self.training):
            return

        with torch.no_grad():
            key_flat = new_key.detach().reshape(-1, new_key.size(-1))
            value_flat = new_value.detach().reshape(-1, new_value.size(-1))
            for i in range(key_flat.size(0)):
                ptr = int(self.rag_memory_ptr.item())
                self.rag_memory_key[ptr].copy_(key_flat[i])
                self.rag_memory_value[ptr].copy_(value_flat[i])
                self.rag_memory_ptr[0] = (self.rag_memory_ptr[0] + 1) % self.rag_memory_capacity
                if self.rag_memory_count[0] < self.rag_memory_capacity:
                    self.rag_memory_count[0] += 1

    def forward(
        self,
        audio_feature,
        video_feature,
        inv_audio_feature,
        inv_video_feature,
        audio_len,
        video_len,
        static_feature=None,
    ):
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
                self._update_rag_memory(torch.cat([new_fwd_key, new_bwd_key], dim=0), torch.cat([new_fwd_value, new_bwd_value], dim=0))

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

            if self.use_static_branch:
                if static_feature is None:
                    raise ValueError("static_feature must be provided when use_static_branch=True")
                static_time_feat = static_feature[i:i+1, :curr_batch_len]  # [1, T, static_in_dim]
                static_time_feat = self.static_proj(static_time_feat)  # [1, T, dim]
                time_feat = torch.cat([time_feat, static_time_feat], dim=-1)  # [1, T, 2*dim]

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
        rag_context = self._read_rag_memory(query)
        if self.training and self.rag_memory_dropout > 0:
            rag_context = nn.functional.dropout(rag_context, p=self.rag_memory_dropout, training=True)
        gate = self.rag_gate(torch.cat([memory_context, rag_context], dim=-1))
        memory_context = gate * rag_context + (1 - gate) * memory_context

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
