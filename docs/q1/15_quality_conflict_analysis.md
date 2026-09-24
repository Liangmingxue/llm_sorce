# Step 15：质量信号冲突的显式定义、成因与扩展集验证

问题一不仅要求给出综合质量分数 Q，还要求显式定义不同质量信号之间的显著冲突、分析成因，并在扩展集上检查主要结论是否保持。

## 1. 冲突强度

Step 7 已经为每条记录输出：

\[
D_i
=
\sum_{j\in\mathcal V_i}
\bar w_{ij}|u_{ij}-Q_i|,
\]

即 `weighted_abs_deviation`。

其中 \(Q_i\) 是最终 Huber 稳健位置，\(u_{ij}\) 为统一到 [0,1] 且越高越好的 22 个指标分数。

因此 \(D_i\) 直接衡量“22 个质量指标围绕最终共识 Q 的加权不一致程度”。

正式定义强冲突为：

\[
D_i\ge \tau_{0.90},
\]

其中 \(\tau_{0.90}\) 是 A1 上 \(D_i\) 的 90% 分位数。

阈值只在 A1 标定一次，并原样应用到 A2/A3；同时报告 q85/q90/q95 敏感性，避免结论依赖单一阈值。

## 2. 为什么使用 A1 分位阈值

这一定义不是声称存在一个自然物理常数式的“冲突边界”，而是给出一个可重复、可校准的强冲突 operational definition：

> 在抽样基准 A1 中，属于最高 10% 加权不一致程度的记录。

这样既能稳定获得足够样本用于成因统计，也能把阈值冻结后检验扩展集是否出现系统性变化。

## 3. 成因解释

22 个指标沿用质量模型的四组结构：

- semantic_quality；
- dsir；
- length_structure；
- rps_other。

对每条记录计算组内加权 composite。强冲突记录的：

\[
g_i^+=\arg\max_g G_{ig},
\qquad
g_i^-=\arg\min_g G_{ig},
\]

形成主要冲突对 \((g_i^+,g_i^-)\)。

同时对每个单指标计算：

\[
w_j
\left[
E(|u_{ij}-Q_i|\mid conflict)
-
E(|u_{ij}-Q_i|\mid nonconflict)
\right],
\]

作为 indicator-level driver score。

## 4. 冲突消解

冲突不是通过删除记录或删除指标处理，而是由 Step 7 的加权 Huber M-estimator 消解：

\[
Q_i
=
\arg\min_q
\sum_j
\bar w_{ij}\rho_\delta(u_{ij}-q).
\]

当某些指标与共识偏差超过 Huber 阈值时，其边际影响被截断，因此极端矛盾信号不会线性拖动最终 Q。

## 5. 扩展集验证

阈值固定后比较：

- A1 arxiv sample vs A2 added-only / full；
- A1 github sample vs A3 added-only / full。

对 conflict rate 使用 2000 次 bootstrap 95% CI。对于 full expanded，sample 和 added-only 分别重采样后按固定观察样本数重构 full rate，以保留已知的子集关系。

重叠记录必须得到完全相同的 conflict flag，否则脚本直接报错。

这样可回答：

> A1 中观察到的冲突结构，在扩展数据上是否仍然成立？
