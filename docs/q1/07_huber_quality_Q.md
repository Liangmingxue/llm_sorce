# Step 7：Huber 冲突消解与记录级综合质量 Q

本步骤在 Step 6 已确定的 22 个 Hierarchical Group-CRITIC 基础权重上，计算每条记录的最终综合质量分数：

\[
Q_i\in[0,1].
\]

## 1. 为什么不直接加权平均

即使 22 个指标已经统一为“越大越好”，单条记录内部仍可能出现明显冲突，例如语义质量较高但结构质量较低。普通加权平均会让极端冲突指标线性影响最终分数，因此使用 Huber M-estimator：

\[
Q_i
=
\arg\min_q
\sum_{j\in\mathcal V_i}
\bar w_{ij}\rho_\delta(u_{ij}-q),
\]

其中 \(\mathcal V_i\) 是第 \(i\) 条记录的有效指标集合，缺失指标不删除整条记录，而是重新归一化：

\[
\bar w_{ij}
=
\frac{w_j}{\sum_{k\in\mathcal V_i}w_k}.
\]

Huber 损失为：

\[
\rho_\delta(r)=
\begin{cases}
\frac12r^2, & |r|\le\delta,\\
\delta|r|-\frac12\delta^2, & |r|>\delta.
\end{cases}
\]

## 2. 阈值标定

阈值只用 A1 标定，A2/A3 不重新估计。

先以每条 A1 记录的加权中位数为稳健中心，得到绝对残差。对 7 个来源域分别计算：

\[
s_d
=
1.4826\cdot
\operatorname{wmed}(|u_{ij}-m_i|),
\]

再采用领域平衡的全局尺度：

\[
s=\operatorname{median}_d(s_d).
\]

最终：

\[
\delta=1.345s.
\]

当前 A1 诊断得到 \(s\approx0.260726\)，因此 \(\delta\approx0.350676\)。

敏感性分析表明，以 \(1.345s\) 为基准，\(1.0s,1.75s,2.0s\) 的 Q 排序 Spearman 均高于 0.9949；说明结果不依赖某一个精确阈值。

## 3. 确定性求解

一维 Huber 位置估计满足：

\[
\sum_j \bar w_{ij}\psi_\delta(u_{ij}-Q_i)=0.
\]

该方程关于 \(Q_i\) 单调，因此正式实现采用确定性二分求根，而不是 IRLS。

验证中，IRLS 与二分解的 Spearman 为 1.0，平均绝对差约 \(1.2\times10^{-11}\)；二分根残差最大约 \(1.3\times10^{-16}\)。这样可以消除少数 IRLS 记录在严格容差下达到迭代上限的歧义。

## 4. 冲突诊断量

除 Q 外，每条记录还输出：

- `Q_weighted_mean`：普通加权平均，仅作对照；
- `huber_adjustment`：Huber Q 与普通加权平均之差；
- `valid_indicator_count`：有效指标数；
- `available_base_weight`：有效指标的原始基础权重和；
- `base_weight_beyond_delta`：落在 Huber 阈值外的基础权重占比；
- `huber_weight_discount`：Huber 实际折扣的总权重；
- `weighted_abs_deviation`：围绕最终 Q 的加权绝对偏差；
- `huber_root_residual`：二分解的一阶条件残差。

## 5. 输出

写入 `artifacts/q1/quality/`：

- `a1_quality_Q.csv.gz`
- `a2_quality_Q.csv.gz`
- `a3_quality_Q.csv.gz`
- `quality_union_Q_dedup.csv.gz`
- `huber_calibration_domain.csv`
- `quality_Q_audit.csv`
- `quality_Q_manifest.json`

本步骤只生成记录级 Q。下一步再基于这些 Q 计算各数据域质量、Bootstrap 置信区间，以及 A1 抽样集与 A2/A3 扩展集的比较。
