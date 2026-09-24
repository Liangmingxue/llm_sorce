# Step 8：领域质量评价与扩展集比较

本步骤把 Step 7 的记录级综合质量 \(Q_i\) 汇总到来源域层面，并回答第 1 问中“各领域质量如何、扩展集相较原抽样集是否发生变化”的要求。

## 1. 领域质量定义

记录级 \(Q_i\) 已经过 22 指标方向统一、分层 CRITIC 去冗余以及 Huber 冲突消解，因此领域级质量采用透明的算术平均：

\[
Q_d=\frac{1}{n_d}\sum_{i\in d}Q_i.
\]

同时报告 median、标准差和 P05/P95，避免只依赖一个均值。

最终七域表中：

- arXiv 使用 A2 扩展全集；
- GitHub 使用 A3 扩展全集；
- book、c4、commoncrawl、stackexchange、wikipedia 使用 A1。

这样既使用了题目提供的扩展数据，又不重复计算 A1 中已经包含于 A2/A3 的记录。

## 2. Bootstrap 95% 置信区间

每个领域均值使用非参数 percentile bootstrap：

\[
Q_d^{*(b)}
=
\frac{1}{n_d}
\sum_{i=1}^{n_d} Q_{I_i^{(b)}},
\qquad b=1,\dots,B.
\]

默认 \(B=2000\)，固定随机种子以保证可复现，95% CI 取 bootstrap 分布的 2.5% 和 97.5% 分位数。

## 3. 为什么扩展集不能与 A1 当作两个独立样本

A2 包含 A1 的全部 1,419 条 arXiv 记录，A3 包含 A1 的全部 10,000 条 GitHub 记录。因此不能把：

\[
A1_{arXiv}\quad\text{vs}\quad A2_{full}
\]

或：

\[
A1_{GitHub}\quad\text{vs}\quad A3_{full}
\]

机械地当成两个独立样本。

正式比较将扩展全集拆成：

\[
\text{expanded full}
=
\text{A1 sample}
\cup
\text{added-only}.
\]

Bootstrap 时分别对 sample 和 added-only 独立重采样，再按照观察到的固定样本量组合：

\[
\bar Q_{full}^{*}
=
\frac{
n_s\bar Q_s^{*}
+
n_a\bar Q_a^{*}
}{
n_s+n_a
}.
\]

这样在估计 \(Q_{full}-Q_s\) 的置信区间时保留了真实的“子集包含关系”。

## 4. 输出

写入 `artifacts/q1/domain_quality/`：

- `a1_domain_quality.csv`：A1 七个来源域的基线质量；
- `expansion_quality_comparison.csv`：arXiv/GitHub 的 sample、added-only、expanded-full 比较及 Bootstrap CI；
- `final_domain_quality.csv`：第 1 问最终七域质量表；
- `domain_quality_manifest.json`：统计口径、随机种子及扩展集处理说明。

扩展比较同时报告原始均值差与标准化均值差。标准化差异只用于衡量变化量级，不取代原始 \(Q\) 差异。
