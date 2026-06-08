# Refactor Plan: Remove Redundant Manual Pooling & Concatenation

## 核心判断

当前 `forward()` 里手工拼接的四项信息，在开启 Transformer encoder 的情况下：

| 特征 | Transformer 是否已覆盖 | 结论 |
|------|----------------------|------|
| `g_pool`（均值池化） | **是**。self-attention 让每个原子已经 attend 到同 group 所有原子，输出嵌入隐含了全局信息 | **移除** |
| `react_edge`（反应物键级） | **是**。GINEConv 通过 edge_attr 把键级传入原子嵌入，Transformer 进一步传播 | **移除** |
| `branch`（反应分支） | **否**。branch 是反应级别的条件，没有进入 encoder，Transformer 不知道 | **保留** |
| `third_body`（第三体标志） | **否**。pressure 条件同样没有进入 encoder | **保留** |

---

## 改动范围

### 唯一改动：`forward()` 删除两段手工拼接

**当前代码（第 495-500 行）**：

```python
# 删除这段
if self.cfg.molecule_balanced_pool:
    g_edge = self._edge_group_pooled_feature(x=x, data=data, edge_group=edge_group)
    pair_feat = torch.cat([pair_feat, g_edge], dim=-1)       # [E, 192]

# 删除这行
pair_feat = torch.cat([pair_feat, data.react_edge.view(-1, 1)], dim=-1)  # [E, 193]
```

**保留不动**：

```python
# 保留 branch（Transformer 不知道反应分支）
if self.cfg.use_branch_feature:
    b = self._contextual_branch_feature(...)
    pair_feat = torch.cat([pair_feat, b], dim=-1)            # +16

# 保留 third_body（Transformer 不知道压力条件）
if self.cfg.use_third_body_feature:
    t = self._edge_group_scalar_feature(...)
    pair_feat = torch.cat([pair_feat, t], dim=-1)            # +1
```

---

## pair_feat 维度变化

```
改前：[h_i || h_j || g_pool || react_edge || branch || third_body]
       64  +  64  +   64   +     1      +   16    +    1     = 210 维

改后：[h_i || h_j || branch || third_body]
       64  +  64  +   16   +    1      = 145 维
```

---

## 需要同步更新的地方

### 1. `__init__` 里的 `in_dim` 计算

```python
# 当前
in_dim = cfg.hidden * 2 + 1           # h_i + h_j + react_edge = 129
if cfg.molecule_balanced_pool:
    in_dim += cfg.hidden               # + g_pool → 193
if cfg.use_branch_feature:
    in_dim += 1 or cfg.branch_context_dim
if cfg.use_third_body_feature:
    in_dim += 1

# 改后
in_dim = cfg.hidden * 2               # h_i + h_j = 128
if cfg.use_branch_feature:
    if cfg.branch_feature_mode == "scalar":
        in_dim += 1
    else:
        in_dim += cfg.branch_context_dim
if cfg.use_third_body_feature:
    in_dim += 1
```

### 2. `rate_mlp` 的 `rate_in_dim` 计算（`__init__` 里）

`rate_mlp` 的输入来自 group 池化，不受 pair_feat 维度影响，**不需要改**。

### 3. `ModelConfig` 里的 `molecule_balanced_pool` 字段

字段本身不需要删除，但它在 `forward()` 里的逻辑会被移除，实际上变成无效配置。
可以在注释里标注 deprecated，留到下一版清理。

---

## 不受影响的部分

- `encode()`：完全不动
- `transformer_decoder.py`：edge decoder 的输入维度由 `in_dim` 决定，会自动适配
- `train.py` / `validate.py`：不需要改
- `reaction_dataset_prediction.py`：不需要改（`MOLECULE_BALANCED_POOL=True` 的配置项失效但不报错）
- `use_latent_branching`（CVAE）：保持不变

---

## 实施步骤

1. 修改 `__init__` 里的 `in_dim` 计算，移除 `molecule_balanced_pool` 和 `react_edge` 的维度贡献
2. 修改 `forward()`，删除 `g_pool` 和 `react_edge` 的两段拼接
3. 跑一次训练，对比改前改后的 eval bond accuracy 和 loss 曲线
4. 确认效果后，在 `ModelConfig` 里将 `molecule_balanced_pool` 标注为 deprecated

## Configuration rollout

为了保留原始行为，实际实现通过 `ModelConfig.edge_decoder_input_mode` 控制：

- `"legacy"`：保留旧的 `[h_i || h_j || g_pool || react_edge || branch || third_body]` 输入
- `"transformer_minimal"`：启用本计划的精简输入 `[h_i || h_j || branch || third_body]`
