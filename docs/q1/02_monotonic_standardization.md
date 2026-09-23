# Step 2：单调质量指标方向统一

本步骤只处理配置文件中已经明确为单调的指标：

- positive：7 个
- negative：2 个

共 9 个指标。

## 统一尺度

为了让 A1、A2、A3 可直接比较，统一使用 A1 的经验分布作为参考尺度。

对正向指标：

\[
u_j(x)=\widehat F^{A1}_j(x)
\]

对负向指标：

\[
u_j(x)=1-\widehat F^{A1}_j(x)
\]

其中 \(\widehat F^{A1}_j\) 为 A1 中第 \(j\) 个指标的经验分布函数（ECDF）。

转换后：

\[
0\le u_j\le1,\qquad u_j\uparrow \Rightarrow \text{质量更高}.
\]

A2、A3 不单独重新拟合尺度，而是复用 A1 的 ECDF，以保证不同数据集的分数口径一致。

## 本步骤暂不处理

- non_monotonic
- domain_dependent
- 最终综合质量 Q
- Huber 冲突消解

## 输出

写入：

`artifacts/q1/standardized/`

包括：

- `a1_monotonic_scores.csv.gz`
- `a2_monotonic_scores.csv.gz`
- `a3_monotonic_scores.csv.gz`
- `monotonic_transform_audit.csv`
- `monotonic_transform_rules.csv`
