# from librosa.core import audio
# from numpy.ma import clip
from cv2 import transform
from torch import nn
from functools import partial
# from einops.layers.torch import Rearrange, Reduce
import torch
import time
import math


class QuerySupportCrossAttention(nn.Module):
    def __init__(self, feature_dim=768, top_classes=2, top_k=4, attention_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.top_classes = int(top_classes)
        self.top_k = int(top_k)
        self.cache_dim = (
            self.feature_dim
            + self.top_classes * self.top_k * self.feature_dim
            + self.top_classes
        )
        self.query_proj = nn.Linear(self.feature_dim, int(attention_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=int(attention_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            kdim=self.feature_dim,
            vdim=self.feature_dim,
            batch_first=True,
        )
        self.output_proj = nn.Linear(int(attention_dim), self.feature_dim)
        self.output_norm = nn.LayerNorm(self.feature_dim)

    def forward(self, static_feature):
        if static_feature.shape[-1] != self.cache_dim:
            raise ValueError(
                "cross-attention cache must have last dimension {}".format(self.cache_dim)
            )
        leading_shape = static_feature.shape[:-1]
        flat = static_feature.reshape(-1, self.cache_dim)
        support_end = self.feature_dim + (
            self.top_classes * self.top_k * self.feature_dim
        )
        query = flat[:, : self.feature_dim]
        supports = flat[:, self.feature_dim : support_end].reshape(
            -1, self.top_classes, self.top_k, self.feature_dim
        )
        class_weights = flat[:, support_end:].clamp_min(0.0)
        class_weights = class_weights / class_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)

        projected_query = self.query_proj(query)
        class_queries = projected_query[:, None, :].expand(
            -1, self.top_classes, -1
        ).reshape(-1, 1, projected_query.shape[-1])
        class_supports = supports.reshape(-1, self.top_k, self.feature_dim)
        class_context, _ = self.cross_attention(
            class_queries, class_supports, class_supports, need_weights=False
        )
        class_context = class_context.reshape(
            -1, self.top_classes, projected_query.shape[-1]
        )
        context = torch.sum(class_context * class_weights[..., None], dim=1)
        output = self.output_norm(query + self.output_proj(context))
        return output.reshape(*leading_shape, self.feature_dim)


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
    def __init__(self, depth, input_dim, dim, input_len=2, num_scores=1, use_static_branch=False, static_in_dim=2048, static_proj_dim=128, use_top4_cross_attention=False, top_classes=2, top_k=4, cross_attention_dim=128, cross_attention_heads=4):
        super().__init__()
        self.use_static_branch = use_static_branch


        self.hidden_state = nn.parameter.Parameter(torch.randn(1, dim))
        self.cls_token = nn.parameter.Parameter(torch.randn(1, dim))

        self.linear1 = nn.Linear(input_dim, dim)
        
        self.linear_forward = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward((input_len + 2), dense = partial(nn.Conv1d, kernel_size=1))),  # token mixing
                PreNormResidual(dim, FeedForward(dim))) for _ in range(depth)]    # channel mixing
        )

        self.layer_norm = nn.LayerNorm(dim)

        self.hidden_linear = nn.Linear(dim, dim)
        self.output = head(dim, num_scores)

        # time-wise scoring head: 2 hidden layers MLP applied on each timestep feature
        # input: [B, T, dim] -> output: [B, T, num_scores]
        hidden1 = dim // 2
        hidden2 = max(dim // 4, 1)
        fused_dim = dim + static_proj_dim if self.use_static_branch else dim
        if self.use_static_branch:
            self.use_top4_cross_attention = bool(use_top4_cross_attention)
            if self.use_top4_cross_attention:
                denominator = 1 + int(top_classes) * int(top_k)
                feature_dim, remainder = divmod(
                    static_in_dim - int(top_classes), denominator
                )
                if remainder or feature_dim <= 0:
                    raise ValueError("static input dimension does not match cross-attention layout")
                self.query_support_cross_attention = QuerySupportCrossAttention(
                    feature_dim=feature_dim,
                    top_classes=top_classes,
                    top_k=top_k,
                    attention_dim=cross_attention_dim,
                    num_heads=cross_attention_heads,
                )
                static_proj_in_dim = feature_dim
            else:
                static_proj_in_dim = static_in_dim
            self.static_proj = nn.Linear(static_proj_in_dim, static_proj_dim)
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

    def forward(self, audio_feature, video_feature, inv_audio_feature, inv_video_feature, audio_len, video_len, static_feature=None):
        batch_size, aclip, _, _ = audio_feature.shape
        batch_size, vclip, _, _ = video_feature.shape
        clip = min(aclip, vclip)

        hidden_states = []
        back_hidden_states = []

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
            
            if j == 0:
                hidden_state, cls = self.model_forward(input_feature, first_frame=True)
                back_hidden_state, back_cls = self.model_forward(back_input_feature, first_frame=True, back=True)

                hidden_states.append(cls)
                back_hidden_states.insert(0, back_cls)

            else:
                hidden_state, cls = self.model_forward(input_feature, hidden_state)
                back_hidden_state, back_cls = self.model_forward(back_input_feature, back_hidden_state, back=True)

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
                static_time_feat = static_feature[i:i+1, :curr_batch_len]
                if self.use_top4_cross_attention:
                    static_time_feat = self.query_support_cross_attention(static_time_feat)
                static_time_feat = self.static_proj(static_time_feat)
                time_feat = torch.cat([time_feat, static_time_feat], dim=-1)

            # MLP predicts score per timestep, then aggregate to scalar score
            time_score = self.time_score_mlp(time_feat).squeeze(-1)  # [1, T]
            batch_score = torch.mean(time_score, dim=1)  # [1]
            final_scores.append(batch_score)

        output = torch.cat(final_scores, dim=0)  # [B]
        return output
    
    def model_forward(self, x, hidden_state=None, first_frame=False, back=False):
        # x shape: B x 2 (a & v) x D

        x = self.linear1(x)

        if back:
            batch_size = x.shape[0]
            if first_frame:
                
                back_hidden_state = self.hidden_state.unsqueeze(dim=0)
                hidden_state = torch.cat([back_hidden_state for _ in range(batch_size)], dim=0)

            back_cls_token = self.cls_token.unsqueeze(dim=0)
            cls_token = torch.cat([back_cls_token for _ in range(batch_size)], dim=0)

            concat_input = torch.cat([cls_token, x, hidden_state], dim=1)

        else:
            batch_size = x.shape[0]
            if first_frame:
                
                hidden_state = self.hidden_state.unsqueeze(dim=0)
                hidden_state = torch.cat([hidden_state for _ in range(batch_size)], dim=0)

            cls_token = self.cls_token.unsqueeze(dim=0)
            cls_token = torch.cat([cls_token for _ in range(batch_size)], dim=0)

            concat_input = torch.cat([hidden_state, x, cls_token], dim=1)
        
        out = self.linear_forward(concat_input)

        if back:
            out_cls = out[:, 0:1]
            out_hs = out[:, -1:]
        else:
            out_hs = out[:, 0:1]
            out_cls = out[:, -1:]

        out_cls = self.hidden_linear(out_cls)

        return out_hs, out_cls


class head(nn.Module):
    def __init__(self, dim, num_scores=1):
        super().__init__()
        self.linear = nn.Linear(dim, num_scores)

    def forward(self, x):
        x = self.linear(x)
        return x
