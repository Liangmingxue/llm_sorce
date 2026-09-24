# Step 9：RegMix 建模前审计

本步骤只审计 A4–A15，不训练模型。

## 核心结论

17 个训练配比满足单纯形约束：

\[
p_j\ge 0,\qquad \sum_{j=1}^{17}p_j=1.
\]

A4/A5 的中心化配比矩阵秩为 16，因此有效自由度为 16，而不是 17。

同时，训练配比中存在大量精确零值。不同域的零占比约为 30%–69%。因此未经处理的 CLR/ILR：

\[
\log\frac{p_j}{g(p)}
\]

在这些样本上并没有定义。后续不能把 ILR 当作默认唯一方案；若比较 log-ratio 模型，必须显式加入 zero-replacement，并做 pseudocount 敏感性分析。

## Recipe overlap

审计确认：

- test_1m 与 test_60m 使用完全相同的 256 个配方，因此是“相同 composition design、不同模型规模”的配对集合；
- test_1B 的 64 个配方与上述集合及 train_1m 均不重叠；
- est_10b 与 est_70b 使用完全相同的 63 个配方；
- 这 63 个配方全部存在于 train_1m。

因此：

- A6–A11 是独立测试，不进入训练；
- A12–A15 是新规模下的估算/外推集合，不计入独立测试指标；
- A12/A14 与 A4 的配方重叠不是数据泄漏，因为它们不会参与模型拟合；它们的用途是检查同一配方跨规模的外推表现。

## 后续建模策略

下一阶段应比较至少两条路线：

1. **Simplex-native regularized polynomial model**：直接在 \(p\) 上建模，但通过去掉一个基准坐标或使用显式约束解决单纯形共线性，并对二阶项做 Ridge/ElasticNet；
2. **Zero-replaced ILR + Ridge/ElasticNet**：作为 compositional-data 分析基线，并对 zero replacement 参数做敏感性分析。

模型选择只允许在 A4/A5 内部进行交叉验证；A6–A11 保留为真正外部验证。
