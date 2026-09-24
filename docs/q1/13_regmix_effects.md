# Step 13：17 域配比效应与组合效应解释

题目要求不仅建立配比与 Loss 的定量关系，还要分析各领域及其组合对性能的影响。

由于配比位于单纯形：

\[
p_j\ge 0,
\qquad
\sum_{j=1}^{17}p_j=1,
\]

不能把某一维“单独增加”而保持其他 16 维完全不变。因此本步骤使用显式质量守恒的 compositional perturbation，而不是普通的逐变量偏导。

## 1. 综合 Loss 目标

13 个验证 Loss 的数值尺度不同，直接求平均会让高方差 target 获得更大权重。

因此先用 A4/A5 的训练均值和标准差标准化：

\[
Z_d(p)
=
\frac{\widehat L_d(p)-\mu_d}{\sigma_d},
\]

再定义等权综合目标：

\[
J(p)=\frac1{13}\sum_{d=1}^{13}Z_d(p).
\]

\(J\) 越低越好。

这不是唯一的业务价值函数，因此后续最优配比步骤还应做 target-weight sensitivity；本步骤只用它作为统一的解释尺度。

## 2. 单领域 enrichment effect

将领域 \(j\) 增加 \(\delta\)，并按原有比例从其余 16 个领域扣除同样总质量：

\[
p'_j=p_j+\delta,
\]

\[
p'_k
=
p_k\frac{1-p_j-\delta}{1-p_j},
\qquad k\ne j.
\]

计算：

\[
\Delta_j(\delta)
=
E[J(p')-J(p)].
\]

- \(\Delta_j<0\)：增加该领域通常降低综合 Loss；
- \(\Delta_j>0\)：增加该领域通常提高综合 Loss。

正式报告 \(\delta=0.01\)，并用 \(0.02\) 检查排序稳定性。

## 3. 两领域替代效应

从 donor \(k\) 精确转移 1% 配比给 receiver \(j\)：

\[
p'_j=p_j+0.01,
\qquad
p'_k=p_k-0.01.
\]

其余分量不变。

这给出最直接的“把哪一类数据换成哪一类数据”解释。

## 4. 两领域组合交互

为避免单纯形约束导致伪交互，对 \(j,k\) 使用相同的 donor pool（剩余 15 域）。

定义：

\[
I_{jk}
=
J(p^{j+k})
-
J(p^j)
-
J(p^k)
+
J(p).
\]

其中单独增加 \(j\)、单独增加 \(k\) 与同时增加 \(j,k\) 都从同一 donor pool 按比例扣除质量。

解释：

- \(I_{jk}<0\)：协同（joint enrichment 的改善超过可加和预期）；
- \(I_{jk}>0\)：拮抗/冗余。

## 5. 不使用测试标签做解释拟合

Step 13 的 reference recipes 只来自 A4/A5，预测器是 Step 10c 已冻结模型。

A6–A11 和 A12–A15 的标签均不参与效应估计。它们只承担前两步已经完成的外部验证和外推稳健性角色。

因此不会出现“看了测试结果以后重新解释/调模型”的信息泄漏。

## 6. 统计不确定性

对 recipe-level finite differences 做 1000 次 bootstrap，输出 95% percentile CI。

这些量应解释为：

> 在冻结预测模型和 A4/A5 的经验配比分布下的 model-based average perturbation effects。

它们不是随机对照实验意义上的因果效应。
