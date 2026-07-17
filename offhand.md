# 下一智能体交接说明

## 目标

在现有 DINOv2 人体动作质量评分项目中，将当前“仅使用 CLS 特征进行分数回归”的实现改为：

```text
CLS 特征 [B, D]
        +
Patch Mean 特征 [B, D]
        ↓ 拼接
融合特征 [B, 2D]
        ↓
128 维隐藏层
        ↓
单个线性回归输出 [B, 1]
```

本次修改不涉及人体区域裁剪，也不涉及任何时序建模。

## 项目运行流程

项目分成两个独立阶段和两个环境：

```text
skating-dinov2 环境
action.py
→ 读取原始数据
→ 运行 DINOv2
→ 提取并保存特征

skating-action 环境
main.py
→ 读取 action.py 保存的特征
→ 执行现有后续处理
→ 通过评分头输出分数
```

因此，本次修改必须分布在两个文件中：

- `action.py`：将原来的 CLS-only 特征改为 `CLS + Patch Mean` 拼接特征，并保存 `[*, 2D]` 特征。
- `main.py`：将模型输入维度改为 `2D`，并将旧三层 MLP 改为“128 维隐藏层 + 单个线性回归输出”。

不要尝试在 `main.py` 中重新运行 DINOv2。两个环境的职责和现有数据传递方式应保持不变。

## 开始前

先定位以下内容：

1. DINOv2 模型初始化位置。
2. 当前提取 `CLS` 特征的位置。
3. 当前三层 MLP 或评分头的定义。
4. 训练代码期望模型输出为 `[B]` 还是 `[B, 1]`。
5. `action.py` 保存特征的文件格式、字段名和形状。
6. `main.py` 加载特征并推断输入维度的位置。

保留现有数据预处理、数据划分、损失函数、优化器以及 DINOv2 的冻结或微调策略。不要借本次修改调整其他实验变量。

## 特征提取

此部分应在 `action.py` 中完成。优先使用 DINOv2 的 `forward_features`，一次前向同时取得 CLS 和 patch tokens：

```python
features = self.backbone.forward_features(images)

cls = features["x_norm_clstoken"]          # [B, D]
patches = features["x_norm_patchtokens"]   # [B, N, D]
patch_mean = patches.mean(dim=1)            # [B, D]

fused = torch.cat([cls, patch_mean], dim=-1)  # [B, 2D]
```

注意事项：

- 只沿 patch/token 维 `dim=1` 求平均。
- 不要把 `[B, N, D]` 展平为 `[B, N*D]`。
- 不要分别运行两次 backbone 来提取 CLS 和 patch。
- 优先使用 `x_norm_clstoken` 和 `x_norm_patchtokens`，确保两类特征来自同一归一化阶段。
- 如果当前 DINOv2 API 只返回完整 token 张量，必须正确排除 CLS token 和可能存在的 register tokens；不要无条件假设 `tokens[:, 1:]` 全部都是 patch tokens。
- 不要新增 `detach()`。是否允许梯度进入 backbone 应继续服从项目现有的冻结策略。

将 `fused` 作为 `action.py` 的新输出特征保存。保持现有样本顺序、样本 ID、标签关联、数据类型和文件结构不变，只替换特征内容及其最后一维大小。

原有 CLS-only 缓存与新模型不兼容。修改后必须在 `skating-dinov2` 环境中重新运行 `action.py` 生成全部特征。不要让 `main.py` 混合读取旧的 `[*, D]` 特征和新的 `[*, 2D]` 特征。若现有流程允许，优先写入新的输出目录或文件名，保留旧基线用于对照。

## 新评分头

此部分应在 `main.py` 中完成。`main.py` 读取的新特征已经是 `[*, 2D]`，无需再次提取或拼接 patch。用以下结构替换当前三层 MLP：

```python
self.score_head = nn.Sequential(
    nn.LayerNorm(2 * feature_dim),
    nn.Linear(2 * feature_dim, 128),
    nn.GELU(),
    nn.Dropout(p=0.3),
    nn.Linear(128, 1),
)
```

其中 `feature_dim` 是 DINOv2 单个 token 的维度。`action.py` 中可从 backbone 获取，例如官方 DINOv2 通常可使用：

```python
feature_dim = self.backbone.embed_dim
```

不要硬编码成 `768`，因为 ViT-S/B/L/g 的特征维度不同。如果项目已有 dropout 配置项，使用现有配置；否则默认 `0.3`。

在 `main.py` 中，评分头的实际输入维度应优先由新特征文件推断，或通过项目配置明确传入。若变量表示“已经拼接后的输入维度”，应直接使用该值，避免再次乘以 2。

如果特征提取和评分处在同一个模型类中，前向传播可参考：

```python
def forward(self, images):
    features = self.backbone.forward_features(images)
    cls = features["x_norm_clstoken"]
    patch_mean = features["x_norm_patchtokens"].mean(dim=1)
    fused = torch.cat([cls, patch_mean], dim=-1)
    score = self.score_head(fused)

    # 仅当现有损失和标签使用 [B] 时保留这一行。
    # score = score.squeeze(-1)
    return score
```

但按本项目当前的两阶段流程，实际实现应拆开：`action.py` 保存 `fused`，`main.py` 的 forward 直接接收已保存的 `fused` 特征并调用 `self.score_head(fused)`。

## 需要删除或更新的旧逻辑

- 删除仅把 `CLS` 送入旧 MLP 的路径。
- 删除旧三层 MLP 中不再使用的层，避免它们仍被优化器统计。
- 如果模型支持通过参数配置输入维度，将评分头输入维度由 `D` 更新为 `2D`。
- 同步更新模型结构日志、配置说明或相关注释，但不要修改无关训练配置。
- 清理或隔离旧 CLS-only 特征缓存，确保训练时加载的是重新生成的拼接特征。

## 验证要求

至少完成以下检查：

1. 用一个小批次执行前向传播，确认 `cls.shape == [B, D]`。
2. 确认 `patches.shape == [B, N, D]`。
3. 确认 `patch_mean.shape == [B, D]`。
4. 确认 `fused.shape == [B, 2D]`。
5. 确认最终输出形状与现有标签和损失函数一致。
6. 执行一次训练步骤，确认前向、损失计算和反向传播均正常。
7. 检查可训练参数，确认旧 MLP 已不再存在，backbone 的冻结状态没有意外改变。
8. 在 `skating-dinov2` 环境中运行一个小规模 `action.py` 预处理样例，确认保存特征的最后一维从 `D` 变为 `2D`。
9. 切换到 `skating-action` 环境，让 `main.py` 读取该样例并执行一次前向和反向传播。
10. 确认两个阶段之间没有维度、数据类型、设备或序列化格式错误。

运行时沿用项目现有命令行参数，不要猜测或改写参数。环境边界应为：

```text
action.py  → skating-dinov2
main.py    → skating-action
```

建议分别记录以下三个基线，便于判断 patch mean 是否真正有效：

```text
CLS only
Patch Mean only
CLS + Patch Mean
```

主实验实现应为第三项。除特征融合和评分头外，其余实验条件保持一致。

## 完成标准

代码最终应实现：

```text
DINOv2 单次前向（action.py / skating-dinov2）
→ normalized CLS 与 normalized patch tokens
→ patch tokens 沿 token 维平均
→ CLS 和 Patch Mean 拼接
→ 保存 2D 特征
→ main.py 在 skating-action 中读取 2D 特征
→ LayerNorm
→ Linear(2D, 128)
→ GELU
→ Dropout(0.3)
→ Linear(128, 1)
```

不要加入人体裁剪、关键点、注意力池化、时序模块或其他额外改动。
