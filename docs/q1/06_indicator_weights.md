# Step 6：指标相关性诊断与初步组平衡 CRITIC 权重

本步骤**不计算最终综合质量 Q**，只解决“22 个指标的信息量与冗余程度如何”的问题。

## 为什么不能直接做普通 CRITIC

A1 的 7 个来源域样本数差异明显。如果直接把全部 A1 行混在一起计算 CRITIC，大样本域会主导权重。

因此这里采用**领域平衡的组内 CRITIC**：

对领域 \(d\)、组 \(g\) 内指标 \(j\)：

\[
C_{dj}
=
\sigma_{dj}
\sum_{k\in g,k\neq j}
\left(1-|\rho_{d,jk}|\right).
\]

其中：

- \(\sigma_{dj}\) 表示指标在领域内的区分度；
- \(|\rho|\) 用于衡量冗余程度；
- 使用绝对相关，是因为强正相关和强负相关都代表强依赖，不能因为负相关就人为增加权重。

再对 7 个领域等权平均：

\[
C_j
=
\frac{1}{7}\sum_d C_{dj}.
\]

组内归一化：

\[
\widetilde w_{j|g}
=
\frac{C_j}{\sum_{k\in g}C_k}.
\]

当前仅为了建立一个可检查的初始权重，4 个指标组暂时等权：

\[
W_g=\frac14.
\]

因此：

\[
w_j^{(0)}
=
\frac14\widetilde w_{j|g}.
\]

这仍然只是**preliminary weight**。下一步还要检查：

1. 高相关指标是否确实被抑制；
2. 领域依赖弱信号是否需要可靠性修正；
3. 最终是否引入 Huber 冲突消解。

## 输出

写入 `artifacts/q1/model/`：

- `indicator_spearman_a1.csv`
- `indicator_correlation_pairs_a1.csv`
- `critic_domain_diagnostics.csv`
- `preliminary_group_balanced_critic_weights.csv`
- `group_weight_summary.csv`
