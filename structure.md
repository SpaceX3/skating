# 花样滑冰自动评分系统 结构说明

## 整体思路

系统目标是预测花样滑冰选手的 **TES（技术得分）**。核心思路是两路并行，最终相加：

```
TES_final = TES_dynamic + Delta_TES_RAG
```

- **TES_dynamic**：基于 FS1000 视频的动态+静态特征，用双向 MRU 直接回归出的基线 TES
- **Delta_TES_RAG**：用 FineFS 外部参考库检索相似动作片段，根据其 GOE 分数推算出的校正残差

---

## 数据集

| 数据集 | 用途 | 规模 |
|--------|------|------|
| **FS1000** | 主训练/验证集，预测 TES | ~1000 个花样滑冰节目视频 |
| **FineFS** | 外部参考库，提供 GOE 语义监督 | 1127 个视频，42143 个动作原型 |

**FineFS 数据划分**（视频不重叠）：
- 训练集：789 个视频（只有这部分进入参考库）
- 验证集：169 个视频（仅作查询）
- 测试集：169 个视频（仅作查询）

**FineFS 动作标签体系**：
- 粗粒度（coarse）：`background / jump / spin / sequence`（4类 + unknown）
- 精细元素（exact element）：242 种（含跳跃组合、旋转变体、步法序列等）

---

## 预处理与特征提取

视频 → 2 秒切片（2 fps）→ DINOv2 提取每帧 CLS+patch-mean 特征（2048维）→ 片内取平均 → 得到每片段的静态特征向量

- 动态特征：TimeSformer + AST 提取的视频/音频帧级特征（时序）
- 静态特征：上述 DINOv2 均值特征（`static_dinov2_cls_patch_mean_cache`）
- FineFS 静态特征存放在 `static_dinov2_cls_patch_mean_rag_cache`，格式为 `.npy`

---

## 模型结构

### 模块一：动态分支（scoring_head）

文件：`model.py`，训练入口：`main.py --training-stage dynamic`

```
[音频帧特征 + 视频帧特征]（正向 + 反向各一路）
        ↓  双向 MRU（Memory Recurrent Unit）
        ↓  前向 CLS + 反向 CLS 取平均 → time_feat [B, T, dim]
        ↓  拼接 DINOv2 静态投影后 time_score_mlp
        ↓  masked_mean → TES_dynamic（标量）
```

双向 MRU 本质是带 cls_token 和 hidden_state 的简化 Transformer 步进单元，逐帧推进后取各步的 CLS 输出拼接为时序特征。

---

### 模块二：语义分类器（SemanticQueryClassifier）

文件：`semantic_rag.py`，训练入口：`semantic_main.py --mode train-classifier`

```
DINOv2静态特征 [B, 2048]
        ↓  LayerNorm → Linear(2048→512) → GELU → Dropout → Linear(512→256)
        ↓  query_token [B, 256]
        ├─ action_head    → action_logit [B]，是否为技术动作
        ├─ coarse_head    → coarse_logits [B, 5]，粗粒度动作类别
        └─ element_head   → element_logits [B, 242]，精细动作元素
```

分类器训练完成后**冻结**，不参与后续阶段的梯度更新。

---

### 模块三：元素条件 GOE 估计（ElementConditionedGOE）

文件：`semantic_rag.py`，训练入口：`semantic_main.py --mode train-goe --classifier-checkpoint <path>`

输入冻结的分类器输出，在 FineFS 参考库中软路由检索，估计 GOE。

**软路由得分**：
```
score(q, r) = α * cosine(z_q, z_r)
            + β_elem  * log P(element_r | q)
            + β_coarse* log P(coarse_r  | q)
            + β_action* log P(action_r  | q)
```
α、β 均用 softplus 约束为非负，可学习。

**GOE 输出路径**：
```
reference_goe（参考片段标注 GOE）
delta_goe    = pair_encoder → delta_goe_head（预测当前查询与参考的 GOE 差）
candidate_goe = reference_goe + delta_goe

evidence_weights = citation_weights × score_valid × P(element_r|q)
evidence_reference_goe = Σ(weights × reference_goe)
evidence_delta_goe     = Σ(weights × delta_goe)
evidence_goe           = evidence_reference_goe + evidence_delta_goe

prior = P(element|q) @ element_goe_prior（元素级 GOE 统计先验）
direct_goe = prior + 3 * tanh(direct_goe_head(query_token))

goe_gate = sigmoid(gate_head) × goe_confidence（置信度控制门）
predicted_goe = gate * evidence_goe + (1-gate) * direct_goe
```

---

### 模块四：TES 残差校正（SemanticEvidenceRAG / EvidenceRAG）

文件：`action_rag.py`，训练入口：`main.py --training-stage rag`

使用预计算的候选文件（`rag_artifacts/candidates_v2/`），对 FS1000 每个时间窗口查询 FineFS 参考库，计算 delta_TES_RAG 并加到动态基线上。

```
[每窗口 DINOv2静态特征 + 动态时序特征]
        ↓  query_encoder → query_embedding
        ↓  候选参考特征 + GOE/BV/panel 元数据 → reference_encoder
        ↓  pair_encoder → citation_head → relative_goe_head
        ↓  evidence 聚合 → local_evidence_head
        ↓  correction_head（初始化为全零）→ delta_TES_RAG（标量）
TES_final = TES_dynamic + delta_TES_RAG
```

correction_head 初始化为零，确保训练初期不破坏动态基线。

---

### 整体流水线示意

```
视频
 ├─ 动态特征(TimeSformer/AST) ──→ scoring_head(MRU) ─────────────────┐
 │                                                                     ↓
 └─ DINOv2静态特征 ─→ SemanticQueryClassifier ─→ ElementConditionedGOE  TES_dynamic
                              (冻结)                  ↓                  ↓
                                             候选GOE文件(candidates_v2)  +
                                                      ↓            Delta_TES_RAG
                                              EvidenceRAG ──────────→ TES_final
```

---

## 训练步骤

所有命令在 `scripts/run_action_rag.sh` 中统一管理。

### 步骤 0：数据预处理
```bash
bash scripts/run_action_rag.sh preprocess-all
```
提取 FineFS 视频的 DINOv2 特征（约数小时，可断点续跑）。

### 步骤 1：构建 FineFS 参考库语料
```bash
bash scripts/run_action_rag.sh semantic-split      # 生成视频不重叠的 70/15/15 划分
bash scripts/run_action_rag.sh train-semantic-classifier  # Stage A：训练分类器
```
**训练目标**：`L = λ_action * BCE + λ_coarse * CE + λ_element * CE`
模型选择依据：验证集 element top-k coverage。

### 步骤 2：冻结分类器，训练 GOE 估计
```bash
CLASSIFIER_CHECKPOINT=<path> bash scripts/run_action_rag.sh train-semantic-goe  # Stage B
```
**训练目标**：
```
L = λ_ret * 多正样本检索NLL + λ_cit * 质量感知引用NLL
  + λ_delta * SmoothL1(delta_goe, target_delta)
  + λ_ev  * SmoothL1(evidence_goe, query_goe)
  + λ_dir * SmoothL1(direct_goe, query_goe)
  + λ_fin * SmoothL1(predicted_goe, query_goe)
```
模型选择依据：验证集 predicted_goe MAE。

### 步骤 3：为 FS1000 生成候选文件
```bash
SEMANTIC_CHECKPOINT=<path> bash scripts/run_action_rag.sh candidates-v2
```
对 FS1000 每个时间窗口，用软路由在 FineFS-train 库中检索，保存 top-k 候选到 `rag_artifacts/candidates_v2/`。

### 步骤 4：训练动态基线
```bash
bash scripts/run_action_rag.sh train-dynamic
```
在 FS1000 上训练 MRU 动态分支，得到 TES_dynamic。

### 步骤 5：训练 RAG 残差校正
```bash
bash scripts/run_action_rag.sh train-rag
```
冻结语义模型，在 FS1000 上训练 EvidenceRAG 的 correction_head。

### 步骤 6：评估
```bash
CHECKPOINT=<rag_best.pth> bash scripts/run_action_rag.sh evaluate
CHECKPOINT=<rag_best.pth> bash scripts/run_action_rag.sh evaluate-dynamic  # 消融：关闭RAG
```

---

## 各阶段当前效果

### 分类器（最佳检查点 epoch 136）

| 指标 | 值 |
|------|----|
| Action 准确率 | 85.0% |
| Coarse top-1 | 65.5% |
| Element top-1 | 23.1% |
| Element top-10 | **78.8%** |

Element top-1 较低（23%），但 top-10 已达 78.8%，软路由保留多候选时可利用较多正确参考。

各粗类情况：jump 最难（top-1 仅 15%），spin 最好（top-1 36%）。

### GOE 估计（最新运行 semantic_goe_seed2026_20260728_162123）

| 分支 | MAE | Spearman |
|------|-----|---------|
| direct_goe | **1.396** | **0.455** |
| evidence_goe | 1.448 | 0.434 |
| predicted_goe（最终） | 1.438 | 0.440 |

检索召回（top-8）：overall ~50%，jump ~37%，sequence ~63%，spin ~63%。

### RAG TES 校正（最新运行 rag_seed2026_20260728_163621）

| | MSE | Spearman |
|--|-----|---------|
| Dynamic 基线（val） | 83.04 | 0.8658 |
| RAG 最佳（val，epoch 69） | **81.33** | **0.8678** |

RAG 带来约 2% 的 MSE 改善；训练集 MSE 约 5.4（过拟合明显）。

---

## 关键文件索引

| 文件 | 职责 |
|------|------|
| `model.py` | `scoring_head`：动态 MRU + RAG 集成的顶层模型 |
| `semantic_rag.py` | `SemanticQueryClassifier`、`ElementConditionedGOE`、`SemanticPipeline` |
| `action_rag.py` | `EvidenceRAG`、`SemanticEvidenceRAG`：TES 残差校正 |
| `semantic_main.py` | FineFS 分类器/GOE 训练循环、评估指标 |
| `main.py` | FS1000 动态分支/RAG 训练循环 |
| `semantic_data.py` | FineFS 数据集加载、视频划分 |
| `dataset/dataset_fs800.py` | FS1000 数据集加载 |
| `scripts/precompute_action_candidates.py` | 生成 rag_artifacts/candidates_v2/ |
| `scripts/run_action_rag.sh` | 所有训练/评估命令的入口 |
| `eval.py` | 独立评估脚本 |

---

## Checkpoint 格式约定

所有 v2 检查点包含 `format_version = "finefs-semantic-v2"`，不兼容 v1 格式（会主动报错）。

- 分类器 checkpoint：含 `training_stage = "finefs_semantic_classifier_v2"` 和 `state_dict`
- GOE checkpoint：含 `training_stage = "finefs_semantic_goe_v2"`、`classifier_state_dict`、`goe_state_dict`
- 候选文件（`.npz`）：含 `format_version`、`semantic_checkpoint_sha256`、`corpus_version` 等元信息