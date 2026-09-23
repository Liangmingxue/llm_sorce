# Step 5：生成完整 22 维统一质量矩阵

本步骤把前面已经完成诊断的 22 个原始质量指标统一转换为：

\[
u_{ij}\in[0,1],\qquad u_{ij}\uparrow \Rightarrow \text{质量/结构可靠性更高}.
\]

注意：这里仍然**不计算最终综合质量 Q**。

## 输出维度

严格保持题目原始 22 个指标的维度和顺序。

QURATER 虽然原始含 4 个子分量，但先分别 ECDF，再等权平均回 1 个 `u_qurater`。

DSIR 三个指标仍输出三个分数；由于它们高度共线，后续综合 Q 时通过“组权重”避免三重计权，而不是在预处理阶段删除指标。

## A1/A2/A3 参考尺度

- 全局型指标：A1 全体样本作为经验分布参考；
- 领域型指标：A1 对应来源域作为参考；
- A2(arXiv)复用 A1-arXiv；
- A3(GitHub)复用 A1-GitHub；
- 不在 A2/A3 上重新拟合阈值或模型。

这样可以保证抽样集与扩展集使用同一评价尺度。

## 特殊处理

### 文本长度

`word_count` 和 `num_sentences` 使用领域内 ECDF，并在 P95 后饱和：

\[
u=\min\left(\frac{F_d(x)}{0.95},1\right).
\]

### Unique-word fraction

先在 A1 各领域拟合：

\[
x=\beta_0+\beta_1\log(1+L)+\beta_2\log^2(1+L)+\epsilon,
\]

再用绝对残差的反向 ECDF：

\[
u=1-F_d(|\epsilon|).
\]

### 典型性指标

对于没有稳定正负方向的领域结构信号：

\[
u_{typ}=1-2|F_d(x)-0.5|.
\]

### 高尾异常

只在诊断显示高尾恶化的字段/领域启用单侧高尾惩罚。

## 输出文件

写入 `artifacts/q1/transformed/`：

- `a1_quality_22_scores.csv.gz`
- `a2_quality_22_scores.csv.gz`
- `a3_quality_22_scores.csv.gz`
- `quality_union_22_scores_dedup.csv.gz`
- `quality_22_transform_audit.csv`
- `quality_22_transform_manifest.json`

其中前三个文件均为：

- 4 个身份/来源字段；
- 22 个统一方向的质量分数。

最终 Q 的权重、组权重、冲突消解和领域聚合属于下一阶段，不在本步骤完成。
