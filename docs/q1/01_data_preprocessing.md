# F题·问题一：数据预处理（Step 1）

本步骤只做数据清洗、结构统一和质量审计，不计算最终质量分 Q，也不训练配比-Loss 模型。

## 输入范围

- A1-A3：质量信号全量记录。
- A4-A15：六对 RegMix 配比/Loss 表。
- A16：17 个配比域到质量域的参考映射。

## 处理规则

### A1-A3 质量信号

1. 使用 `lzma.open(..., "rt")` 流式读取 `.jsonl.xz`。
2. 将 8 个列表型字段转换为可建模标量：
   - `fineweb_edu`：取列表第 1 个值；
   - `fluency_en`：2 类 logits 经 softmax 后取“流畅”概率；
   - `ad_en`：2 类 logits 经 softmax 后取“无广告”概率；
   - 4 个 `modernbert_*`：6 类 logits 经 softmax 后计算 0-5 等级期望，再除以 5；
   - `qurater`：保留 4 个分量，同时给出四分量均值 `qurater`。
3. 非有限值（NaN/Inf）统一保留为缺失值，并写入审计表；本步骤不做静默插补。
4. A1、A2、A3 分开保存；另输出一个按 `id` 去重的 pooled 表，避免 A1 的 arXiv/GitHub 样本与 A2/A3 重复计数。
5. 本步骤不做最终方向统一、分位数归一化或 Huber 综合评分；这些属于下一步质量评价模型。

## A4-A15 RegMix

1. 每一对 mixture/loss 表显式按 `index` 做 one-to-one merge。
2. 原始 17 域配比保留为 `p_raw_*`。
3. 因原始表存在千分位舍入误差，再额外生成严格满足单纯形约束的：
   
   ```
   p_j = p_raw_j / sum_k p_raw_k
   ```

4. 训练、测试、外推三类数据严格分开：
   - A4-A5：train
   - A6-A11：test
   - A12-A15：estimated/extrapolation
5. Loss 字段统一重命名为 `loss_<domain>`。

## 运行

如果 Git LFS 文件尚未拉取：

```bash
git lfs pull
```

安装依赖：

```bash
pip install -r requirements-q1.txt
```

执行：

```bash
python scripts/q1/01_preprocess.py
```

## 主要输出

生成于 `artifacts/q1/preprocessed/`：

- `a1_quality_clean.csv.gz`
- `a2_quality_clean.csv.gz`
- `a3_quality_clean.csv.gz`
- `quality_union_dedup.csv.gz`
- `quality_dataset_audit.csv`
- `quality_field_audit.csv`
- `quality_overlap.json`
- `regmix_train_1m_clean.csv`
- `regmix_test_1m_clean.csv`
- `regmix_test_60m_clean.csv`
- `regmix_test_1B_clean.csv`
- `regmix_est_10b_clean.csv`
- `regmix_est_70b_clean.csv`
- `regmix_dataset_audit.csv`
- `domain_mapping_clean.csv`
- `preprocessing_manifest.json`

## 成功判据

- A1/A2/A3 均成功解析；
- 22 个质量字段均存在；
- A4-A15 六对表均通过 index 一一对应检查；
- 每个 RegMix clean 表的 17 个 `p_*` 行和在数值误差范围内等于 1；
- A6-A11 没有混入训练集；
- A12-A15 保持“外推/估算”标签，不被当作真实验证集。
