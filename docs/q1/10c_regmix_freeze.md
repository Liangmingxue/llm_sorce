# Step 10c：外部测试前冻结最终 RegMix 模型

Step 10 的 nested CV 已经给出目标级模型族选择：

- ILR-Ridge：9 个 loss target；
- Quadratic Ridge Simplex：4 个 loss target。

Step 10b 说明 ILR 的原 epsilon 上界 \(10^{-3}\) 过窄，但它并没有形成一个简单的“epsilon 达到边界就改用 Quadratic Ridge”的规则。例如 USPTO 和 Wikipedia 在较大 epsilon 下仍明显优于 Quadratic Ridge，而 Pile-CC 和 PubMed Abstracts 更适合 Quadratic Ridge。

因此本步骤采用一个严格的冻结原则：

1. **模型族不再改变**：完全沿用 Step 10 nested-CV 的 9/4 target-wise winner；
2. **Step 10b 只扩展 ILR epsilon 的调参网格**，不使用其 outer-OOF 结果重新指定模型族；
3. 所有最终超参数仅在 A4/A5 的 512 个 train_1m 配方上用 5-fold CV 选择；
4. 保存每个 target 的最终 estimator、配置和 SHA256；
5. 生成 `FROZEN_BEFORE_EXTERNAL_TEST` manifest 后，才允许打开 A6–A11。

## 冻结模型族

ILR-Ridge：

- arxiv
- dm_mathematics
- freelaw
- github
- pubmed_central
- stackexchange
- ubuntu_irc
- uspto_backgrounds
- wikipedia_en

Quadratic Ridge Simplex：

- gutenberg_pg_19
- hackernews
- pile_cc
- pubmed_abstracts

## 为什么 Gutenberg 仍保留 Quadratic Ridge

Step 10b 中 Gutenberg 的最佳 ILR NRMSE 与 Quadratic Ridge 几乎相同，差约 \(1.6\times10^{-4}\)。原 Step 10 nested-CV 中 Quadratic Ridge 是该 target 的 winner，因此没有充分理由为了极小的诊断性差异改变冻结模型族。

## 输出

`artifacts/q1/regmix_final/`：

- `models/*.joblib`：13 个最终模型；
- `frozen_target_models.csv`：target、family、最终超参数、CV 指标与模型哈希；
- `final_tuning_cv_results.csv`：完整最终调参轨迹；
- `freeze_manifest.json`：外部测试前冻结清单。

在该 manifest 生成后，A6–A11 的结果只能作为外部验证结果报告，不得用于回头修改模型族或 epsilon/alpha 网格。
