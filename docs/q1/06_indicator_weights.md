# Step 6：分层领域平衡 CRITIC 权重

本步骤仍然**不计算最终综合质量 Q**，而是确定进入稳健聚合前的 22 个基础指标权重。

## 第一层：组内领域平衡 CRITIC

对来源域 \(d\)、指标组 \(g\) 内指标 \(j\)：

\[
C_{dj}
=
\sigma_{dj}
\sum_{k\in g,k\neq j}
\left(1-|\rho_{d,jk}|\right).
\]

使用 \(|\rho|\) 是因为强正相关和强负相关都表示冗余。随后对 7 个来源域等权平均：

\[
C_j=\frac{1}{7}\sum_d C_{dj},
\qquad
\widetilde w_{j|g}
=
\frac{C_j}{\sum_{k\in g} C_k}.
\]

这一步只决定**同一组内部**各指标的相对权重。

## 为什么不能直接四组各 0.25

初步诊断表明：

- 三个 DSIR 高度冗余；
- `word_count` 与 `num_sentences` 高度相关；
- 不同组包含的指标数量不同，直接对组综合分使用原始方差会让“指标少的组”因方差较大而被 CRITIC 人为放大。

此外，DSIR 原始值受到严重文本长度混杂。Step 5 已将 DSIR 改为领域内、word-count 条件化的正向 ECDF，使 DSIR 组与长度结构组的 pooled Spearman 从约 0.899 降至约 -0.085。

因此，四组等权 \(1/4\) 只保留为**诊断基线**，不作为最终基础权重。

## 第二层：组间 CRITIC

先用第一层权重构造组综合分：

\[
G_{ig}
=
\frac{
\sum_{j\in g,\,u_{ij}\text{ available}}
\widetilde w_{j|g}u_{ij}
}{
\sum_{j\in g,\,u_{ij}\text{ available}}
\widetilde w_{j|g}
}.
\]

由于不同组的指标数量和内部冗余程度不同，直接比较 \(G_{ig}\) 的标准差会产生“组规模压缩偏差”。因此在每个来源域内对每个组综合分做 A1 midrank ECDF：

\[
\widetilde G_{ig}
=
\widehat F_{dg}(G_{ig}).
\]

这是单调变换，因此保留组间 Spearman 依赖结构，同时统一组尺度。

随后对 4 个组计算领域平衡 CRITIC：

\[
C_{dg}
=
\sigma_{dg}
\sum_{h\neq g}
\left(1-|\rho_{d,gh}|\right),
\]

\[
C_g
=
\frac{1}{7}\sum_d C_{dg},
\qquad
W_g
=
\frac{C_g}{\sum_h C_h}.
\]

最终第 \(j\) 个指标的基础权重为：

\[
w_j=W_g\widetilde w_{j|g}.
\]

满足：

\[
\sum_{j=1}^{22}w_j=1.
\]

注意这里称为 **redundancy-adjusted CRITIC**：与教科书常见的 \(1-\rho\) 不同，这里使用 \(1-|\rho|\)，因为在已经统一“越大越好”的指标空间中，强负相关同样意味着强依赖和信息重复。

## 输出

写入 `artifacts/q1/model/`：

- `indicator_spearman_a1.csv`：22×22 pooled Spearman；
- `indicator_correlation_pairs_a1.csv`：指标对相关性排序；
- `critic_domain_diagnostics.csv`：第一层领域诊断；
- `preliminary_group_balanced_critic_weights.csv`：四组等权基线，仅用于对照；
- `group_score_spearman_a1.csv`：第一层组综合分的 pooled Spearman；
- `group_critic_domain_diagnostics.csv`：第二层组间 CRITIC 领域诊断；
- `group_critic_weights.csv`：最终 4 个组权重；
- `hierarchical_group_critic_weights.csv`：22 个最终基础权重；
- `group_weight_summary.csv`：基线与最终组权重汇总。

这些权重仍不是最终 Q。下一阶段使用稳健 Huber 聚合处理单条记录内部的指标冲突，并在缺失指标时对有效权重重新归一化。
