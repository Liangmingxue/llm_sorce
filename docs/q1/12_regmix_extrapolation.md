# Step 12：A12–A15 外推稳健性分析

问题一要求 A6–A11 用于检验，并利用 A12–A15 的外推表讨论结论稳健性。

A12–A15 的角色与 A6–A11 不同：它们是估算/外推表，不应表述为独立实验观测。本项目的数据审计还发现：

- 10B 和 70B 各有 63 个配方；
- 两组使用完全相同的 63 个配方；
- 这 63 个配方全部来自 A4/A5 的 1M 训练配方。

因此本步骤不报告“独立测试准确率”，而是考察**配比效应随规模扩展是否保持结构稳定**。

## 1. 三组配对比较

对每个 loss target 比较：

\[
L_{1M}^{obs}(p)
\leftrightarrow
L_{10B}^{est}(p),
\]

\[
L_{1M}^{obs}(p)
\leftrightarrow
L_{70B}^{est}(p),
\]

以及：

\[
L_{10B}^{est}(p)
\leftrightarrow
L_{70B}^{est}(p).
\]

报告：

- Pearson；
- Spearman；
- Kendall；
- affine slope / \(R^2\)；
- centered NRMSE；
- 最优 20% 配方重合率。

其中 loss 越低越好，因此“最优 20%”取 loss 最小的配方。

## 2. 单调性检查

如果 scale 增大，理论上整体 Loss 应降低。因此对每个 target 计算：

\[
P(L_{10B}<L_{1M}),
\]

\[
P(L_{70B}<L_{10B}),
\]

以及：

\[
P(L_{1M}>L_{10B}>L_{70B}).
\]

这是对外推表内部一致性的直接检查。

## 3. 冻结模型与外推表

同时计算冻结的 1M composition model 与 10B/70B 估计 Loss 之间的 rank/affine transfer。

但这 **不是独立测试**，因为这 63 个配方来自训练配方。该结果只能用于说明：

> 在已观察过的 composition design 上，1M 学到的配比响应排序与估计的大规模响应是否一致。

不得把这一结果写成 10B/70B 的真正外部预测精度，也不得根据 A12–A15 回头调模型。

## 4. 与 Step 11 的结合

Step 11 的 A6–A11 是正式独立检验；Step 12 是外推稳健性证据。

论文中两者应分别表述：

- **External validation**：A6–A11；
- **Extrapolation robustness / consistency**：A12–A15。
