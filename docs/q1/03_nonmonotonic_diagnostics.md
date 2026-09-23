# Step 3：非单调指标诊断

本步骤**不直接规定最优区间**，而是先用数据决定后续变换形式。

需要诊断的 6 个指标：

- `rps_doc_word_count`
- `rps_doc_num_sentences`
- `rps_doc_unigram_entropy`
- `rps_doc_frac_unique_words`
- `rps_lines_uppercase_letter_fraction`
- `rps_doc_mean_word_length`

## 为什么先诊断

这些指标不能机械采用“越大越好”或“越小越好”。

例如文本长度过短可能信息不足，但过长也不必然意味着更高质量。因此在定义钟形、区间型或尾部惩罚变换之前，需要先检查：

1. A1/A2/A3 的整体分布；
2. A1 七个来源域的分位数差异；
3. 每个指标不同分位区间与已有单调质量信号之间的关系。

## 临时质量锚点

仅用于诊断，不作为最终 Q。

将 Step 2 得到的 9 个单调质量分数逐样本取中位数：

\[
A_i = \operatorname{median}(u_{i1},\ldots,u_{i9}).
\]

然后把每个非单调指标按经验分位划为 10 档，比较各档的 \(A_i\)。

这样可以判断：

- 是否实际近似单调；
- 是否存在中间最优区间；
- 是否只有极端尾部需要惩罚；
- 是否具有明显领域差异。

## 输出

`artifacts/q1/diagnostics/`

- `nonmonotonic_distribution_audit.csv`：A1/A2/A3 全局分布；
- `nonmonotonic_a1_domain_quantiles.csv`：A1 分领域分位数；
- `nonmonotonic_a1_anchor_deciles.csv`：指标十分位与临时质量锚点关系。
