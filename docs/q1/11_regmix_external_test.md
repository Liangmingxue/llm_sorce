# Step 11：A6–A11 外部检验

问题一要求配比模型在 A6–A11 上验证，A12–A15 留作外推稳健性讨论。

本步骤只读取已经冻结的 13 个模型；任何模型族、epsilon 网格和 Ridge alpha 网格均不得根据外部结果修改。

## 1. 三个检验尺度

- test_1m：256 个未参与训练的配方；
- test_60m：256 个配方，与 test_1m 的 composition design 完全相同；
- test_1B：64 个新的 composition 配方。

## 2. 为什么跨尺度不能只看 raw RMSE

冻结模型拟合的是 1m scale 下的：

\[
p\mapsto L_{1m}(p).
\]

当规模从 1m 变为 60m 或 1B 时，绝对 Loss 水平会发生系统性变化。因此：

- test_1m：直接 RMSE、NRMSE、\(R^2\) 是主要外部预测指标；
- test_60m/test_1B：仍报告 raw direct 指标，但主要观察 mixture-response 是否跨尺度保留，即 Pearson、Spearman 与去均值偏移后的 centered NRMSE。

centered 指标仅去掉：

\[
b=\overline{\hat L-L},
\qquad
\hat L^{(c)}=\hat L-b,
\]

它只用于诊断 composition effect 的形状迁移，不是新的预测模型。

另外报告：

\[
L_{scale}\approx a+b\hat L_{1m}
\]

的 post-hoc affine diagnostic。其参数使用测试标签计算，因此只能描述跨尺度形状，不可作为预测性能或反向调参依据。

## 3. test_1m 与 test_60m 的配对设计

两者使用完全相同的 256 个配方，因此额外直接比较：

\[
L_{1m}(p_i)
\quad\text{vs}\quad
L_{60m}(p_i),
\]

计算 Pearson、Spearman、线性斜率和 affine \(R^2\)，检查真实数据本身的配比效应是否随规模保持。

## 4. 一次性原则

成功运行后生成：

`EXTERNAL_TEST_EVALUATED.json`

再次执行脚本会拒绝运行。这个机制不是技术限制，而是为了把 A6–A11 保持为真正的外部验证集。

A12–A15 在本步骤完全不读取；它们只在下一步用于外推稳健性讨论。
