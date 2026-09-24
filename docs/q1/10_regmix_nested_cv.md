# Step 10：A4/A5 内部 Nested-CV 配比–Loss 模型比较

本步骤第一次进入配比–Loss 建模，但**只允许使用 A4/A5 的 train_1m 512 个配方**。A6–A15 均保持封存，不参与模型选择。

## 1. 为什么使用 Nested CV

若直接在同一组 K-fold 上同时调超参数和报告性能，会对模型选择性能产生乐观偏差。因此使用：

- outer 5-fold：估计模型族的 out-of-fold 泛化误差；
- inner 4-fold：只在每个 outer training fold 内选择超参数。

外部测试数据不会参与任何超参数选择。

## 2. 单纯形坐标

17 个配比满足：

\[
\sum_{j=1}^{17}p_j=1,
\]

因此有效维数为 16。对可以直接处理零值的模型，先使用固定 Helmert 正交基 \(H\in\mathbb R^{17\times16}\)：

\[
z=pH.
\]

这样避免“17 个比例 + 截距”的精确共线性，也避免任意删除某一个参考域。

## 3. 比较的模型

### Mean baseline

只预测 outer training fold 的目标均值，用来判断模型是否真的学到了配比效应。

### Linear simplex

\[
L=\beta_0+\beta^Tz.
\]

### Ridge simplex

在线性 simplex 坐标上加 \(L_2\) 正则化。

### Quadratic Ridge simplex

构造 \(z\) 的所有一次项、平方项和二阶交互项，再使用 Ridge：

\[
L=
\beta_0
+
\sum_j\beta_jz_j
+
\sum_{j\le k}\gamma_{jk}z_jz_k.
\]

由于二阶特征数量远大于 16，一定使用正则化，不能直接普通最小二乘。

### ElasticNet simplex

在线性 simplex 坐标上使用 ElasticNet，检查是否可以通过稀疏化获得更稳定的配比响应。

### Zero-replaced ILR + Ridge

由于原始配比中存在大量精确零值，ILR 不能直接应用。比较以下 epsilon：

\[
10^{-6},10^{-5},10^{-4},5\times10^{-4},10^{-3}.
\]

先做：

\[
\tilde p_j
=
\frac{p_j+\epsilon}
{\sum_k(p_k+\epsilon)},
\]

再计算正交 log-ratio 坐标并用 Ridge。这个模型作为 compositional-data 基线，并显式记录 epsilon 的选择频率。

## 4. 评价指标

每个 13 个 loss target 分别计算 OOF：

- RMSE；
- NRMSE = RMSE / train target standard deviation；
- MAE；
- \(R^2\)；
- Spearman。

模型族总体比较的主要指标是 13 个目标的 macro NRMSE。

## 5. 输出

生成于 `artifacts/q1/regmix_cv/`：

- `nested_cv_target_metrics.csv`
- `nested_cv_fold_metrics.csv`
- `nested_cv_selected_params.csv`
- `nested_cv_oof_predictions.csv.gz`
- `model_family_summary.csv`
- `best_family_by_target.csv`
- `ilr_epsilon_selection.csv`
- `nested_cv_manifest.json`

本步骤不根据外部测试结果反向修改模型。完成内部模型比较后，再冻结候选模型并一次性评估 A6–A11。
