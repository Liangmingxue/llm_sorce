# Step 10b：ILR 零替换参数敏感性

Step 10 的 Nested-CV 表明 ILR-Ridge 在 13 个 loss target 中有 9 个取得最低 NRMSE，并且每个 target 的 epsilon 在 5 个 outer fold 内完全一致。

但原始 epsilon 网格的上界是 \(10^{-3}\)，其中 11/13 个 target 都选择了这个上界。这个现象有两种可能：

1. \(10^{-3}\) 附近确实是稳定区域；
2. 最优点仍在网格外，Step 10 的 ILR 优势部分由人为边界造成。

在打开 A6–A11 之前必须区分这两种情况。

## 方法

只使用 A4/A5 的 512 个 train_1m 配方，不接触任何外部测试集。

固定每一个 epsilon，重新执行与 Step 10 一致的：

- outer 5-fold；
- inner 4-fold；
- inner CV 只调 Ridge alpha。

这样得到每个 epsilon 的真正 OOF 曲线，而不是只统计 GridSearch 选中了几次。

主零替换范围定义为：

\[
\epsilon\le10^{-3}.
\]

同时加入：

\[
1.5\times10^{-3}, 2\times10^{-3}, 5\times10^{-3}, 10^{-2}
\]

作为 stress-test offset。它们的目的不是自动成为最终 zero-replacement 参数，而是检查 Step 10 的最优值是否仍沿着上边界继续改善。

## 判定

- 若 macro NRMSE 在 \(10^{-3}\) 附近形成平台或反弹，且各 target 最优点大多不再落在最大 epsilon，则 ILR 结果可视为稳定；
- 若性能一直随 epsilon 增大且全局最优仍落在最大 stress-test 值，说明“ILR 优势”高度依赖 offset，不应把其解释为稳健的 compositional log-ratio 结果；
- 若某些 target 仍对 epsilon 敏感，可保留 Quadratic Ridge 作为这些目标的候选。

本步骤仍不打开 A6–A11，因此不会产生外部测试泄漏。
