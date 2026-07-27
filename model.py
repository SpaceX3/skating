from functools import partial

import torch
from torch import nn

from action_rag import SemanticEvidenceRAG, masked_mean
from semantic_rag import FineFSSemanticRAG


def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout),
    )


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class scoring_head(nn.Module):
    """Metric-matched FS1000 baseline with an optional evidence-only RAG residual."""

    def __init__(
        self,
        depth,
        input_dim,
        dim,
        input_len=2,
        num_scores=1,
        use_static_branch=False,
        use_static_baseline=False,
        static_in_dim=2048,
        static_proj_dim=256,
        baseline_static_proj_dim=512,
        baseline_head_type="metric",
        time_score_dropout=0.2,
        rag_corpus=None,
        rag_semantic_model=None,
        rag_delta_max=20.0,
    ):
        super().__init__()
        if num_scores != 1:
            raise ValueError("the simplified RAG implementation supports TES only")
        self.dim = int(dim)
        self.use_static_branch = bool(use_static_branch)
        self.use_static_baseline = bool(use_static_baseline)
        self.baseline_head_type = str(baseline_head_type)
        if self.baseline_head_type not in ("metric", "legacy-womean"):
            raise ValueError("unknown baseline_head_type: {}".format(baseline_head_type))

        self.hidden_state = nn.Parameter(torch.randn(1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, dim))
        self.linear1 = nn.Linear(input_dim, dim)
        self.linear_forward = nn.Sequential(
            *[
                nn.Sequential(
                    PreNormResidual(
                        dim,
                        FeedForward(
                            input_len + 2,
                            dense=partial(nn.Conv1d, kernel_size=1),
                        ),
                    ),
                    PreNormResidual(dim, FeedForward(dim)),
                )
                for _ in range(depth)
            ]
        )
        # These names and shapes match the provided legacy FS1000 checkpoint.
        # layer_norm/output are retained because they are present in that state
        # dict, although its time-wise scoring path does not call them.
        self.layer_norm = nn.LayerNorm(dim)
        self.hidden_linear = nn.Linear(dim, dim)
        self.output = head(dim, num_scores)
        hidden_dim = max(dim // 2, 128)
        baseline_input_dim = dim
        if self.use_static_baseline:
            self.static_proj = nn.Linear(static_in_dim, baseline_static_proj_dim)
            baseline_input_dim += baseline_static_proj_dim
        if self.baseline_head_type == "legacy-womean":
            legacy_hidden = max(dim // 4, 1)
            self.time_score_mlp = nn.Sequential(
                nn.Linear(baseline_input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, legacy_hidden),
                nn.GELU(),
                nn.Linear(legacy_hidden, num_scores),
            )
        else:
            self.time_score_mlp = nn.Sequential(
                nn.LayerNorm(baseline_input_dim),
                nn.Linear(baseline_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(time_score_dropout),
                nn.Linear(hidden_dim, num_scores),
            )

        self.rag = None
        if self.use_static_branch:
            if rag_corpus is None:
                raise ValueError("rag_corpus is required when use_static_branch=True")
            if rag_semantic_model is None:
                rag_semantic_model = FineFSSemanticRAG(
                    coarse_classes=len(rag_corpus["coarse_class_vocab"]),
                    elements=len(rag_corpus["element_vocab"]),
                    query_dim=static_in_dim,
                    evidence_dim=static_proj_dim,
                )
            self.rag = SemanticEvidenceRAG(
                corpus=rag_corpus,
                semantic_model=rag_semantic_model,
                dynamic_dim=dim,
                delta_max=rag_delta_max,
            )

    def model_forward(self, x, hidden_state=None, first_frame=False, back=False):
        x = self.linear1(x)
        batch_size = x.shape[0]

        if first_frame:
            hidden_state = self.hidden_state.unsqueeze(0).expand(batch_size, -1, -1)
        cls_token = self.cls_token.unsqueeze(0).expand(batch_size, -1, -1)

        if back:
            concat_input = torch.cat([cls_token, x, hidden_state], dim=1)
        else:
            concat_input = torch.cat([hidden_state, x, cls_token], dim=1)

        out = self.linear_forward(concat_input)
        if back:
            out_cls = out[:, 0:1]
            out_hs = out[:, -1:]
        else:
            out_hs = out[:, 0:1]
            out_cls = out[:, -1:]
        return out_hs, self.hidden_linear(out_cls)

    def encode_dynamic(
        self,
        audio_feature,
        video_feature,
        inv_audio_feature,
        inv_video_feature,
        audio_len,
        video_len,
    ):
        batch_size, audio_steps = audio_feature.shape[:2]
        video_steps = video_feature.shape[1]
        if len(audio_len) != batch_size or len(video_len) != batch_size:
            raise ValueError("audio_len/video_len must contain one value per sample")
        length_values = [min(int(a), int(v)) for a, v in zip(audio_len, video_len)]
        if any(length < 0 for length in length_values):
            raise ValueError("dynamic feature lengths cannot be negative")
        if any(length > audio_steps or length > video_steps for length in length_values):
            raise ValueError("dynamic feature length exceeds the padded tensor")
        steps = max(length_values, default=0)
        if steps <= 0:
            raise ValueError("dynamic feature sequence is empty")

        forward_cls = []
        backward_cls = []
        hidden_state = None
        back_hidden_state = None
        for step in range(steps):
            current = torch.cat(
                [audio_feature[:, step], video_feature[:, step]], dim=1
            )
            current_back = torch.cat(
                [inv_audio_feature[:, step], inv_video_feature[:, step]], dim=1
            )
            hidden_state, cls = self.model_forward(
                current, hidden_state, first_frame=(step == 0)
            )
            back_hidden_state, back_cls = self.model_forward(
                current_back,
                back_hidden_state,
                first_frame=(step == 0),
                back=True,
            )
            forward_cls.append(cls)
            backward_cls.append(back_cls)

        forward_time = torch.cat(forward_cls, dim=1)
        backward_generated = torch.cat(backward_cls, dim=1)
        backward_time = torch.zeros_like(backward_generated)
        for row, length in enumerate(length_values):
            backward_time[row, :length] = torch.flip(
                backward_generated[row, :length], dims=[0]
            )
        dynamic_time_feat = (forward_time + backward_time) / 2.0

        lengths = torch.as_tensor(
            length_values,
            device=dynamic_time_feat.device,
            dtype=torch.long,
        )
        time_index = torch.arange(steps, device=dynamic_time_feat.device)
        dynamic_valid_mask = time_index.unsqueeze(0) < lengths.unsqueeze(1)
        dynamic_time_feat = dynamic_time_feat * dynamic_valid_mask.unsqueeze(-1).to(
            dynamic_time_feat.dtype
        )
        return dynamic_time_feat, dynamic_valid_mask

    def forward(
        self,
        audio_feature,
        video_feature,
        inv_audio_feature,
        inv_video_feature,
        audio_len,
        video_len,
        static_feature=None,
        static_valid_mask=None,
        candidate_indices=None,
        candidate_similarities=None,
        overlap_weights=None,
        return_dict=True,
    ):
        dynamic_time_feat, dynamic_valid_mask = self.encode_dynamic(
            audio_feature,
            video_feature,
            inv_audio_feature,
            inv_video_feature,
            audio_len,
            video_len,
        )
        baseline_time_feat = dynamic_time_feat
        if self.use_static_baseline:
            if static_feature is None:
                raise ValueError("static_feature is required by the FS1000 baseline")
            if static_feature.shape[:2] != dynamic_time_feat.shape[:2]:
                raise ValueError(
                    "baseline static/dynamic features must share [B,T], got "
                    "{} and {}".format(
                        tuple(static_feature.shape[:2]),
                        tuple(dynamic_time_feat.shape[:2]),
                    )
                )
            baseline_time_feat = torch.cat(
                [dynamic_time_feat, self.static_proj(static_feature)], dim=-1
            )
        baseline_time_score = self.time_score_mlp(baseline_time_feat).squeeze(-1)
        tes_baseline = masked_mean(
            baseline_time_score, dynamic_valid_mask, dim=1
        )

        output = {
            "score": tes_baseline,
            "tes_baseline": tes_baseline,
            # Backward-compatible alias for previously generated result readers.
            "tes_dynamic": tes_baseline,
            "delta_tes_rag": torch.zeros_like(tes_baseline),
            "dynamic_time_feat": dynamic_time_feat,
            "dynamic_valid_mask": dynamic_valid_mask,
        }
        if self.rag is not None:
            required = {
                "static_feature": static_feature,
                "static_valid_mask": static_valid_mask,
                "candidate_indices": candidate_indices,
                "candidate_similarities": candidate_similarities,
                "overlap_weights": overlap_weights,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError("RAG inputs are missing: {}".format(missing))
            rag_output = self.rag(
                dynamic_time_feat=dynamic_time_feat,
                static_raw=static_feature,
                dynamic_valid_mask=dynamic_valid_mask,
                static_valid_mask=static_valid_mask,
                candidate_indices=candidate_indices,
                candidate_similarities=candidate_similarities,
                overlap_weights=overlap_weights,
                tes_baseline=tes_baseline,
            )
            output.update(rag_output)
            output["score"] = tes_baseline + rag_output["delta_tes_rag"]

        if return_dict:
            return output
        return output["score"]


class head(nn.Module):
    def __init__(self, dim, num_scores=1):
        super().__init__()
        self.linear = nn.Linear(dim, num_scores)

    def forward(self, x):
        return self.linear(x)
