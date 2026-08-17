# v12：Transformer — 抛弃循环，全靠注意力

> 你将学到：self-attention；多头注意力；位置编码；transformer 完整结构。

## 梗概

终点站。RNN 的两个痛（v10 埋伏笔：顺序处理、梯度消失），attention 一次性解决：**每个词同时看所有词**（并行）、**没有距离衰减**。v11 的「跨序列注意力」升级成「序列内的 self-attention」。这就是「Attention is All You Need」，也是 GPT 系列的架构本体。

## 大纲

### 1. 回顾：为什么抛弃 RNN
顺序处理（慢、难并行）+ 长程依赖（忘事）。attention 让每个位置直接看到所有位置。

### 2. self-attention：每个词看所有词
`softmax(QKᵀ/√dₖ)·V`。Query/Key/Value 三投影，一个位置对全序列求注意力权重、加权求和。

### 3. QKV 是什么
Q（我在找什么）、K（我能提供什么）、V（我的内容）。`Q·K` 算相关性，softmax 归一化，`·V` 加权聚合。

### 4. 多头注意力
多组 QKV 并行，各自关注不同关系（语法、语义、指代…），最后拼接。

### 5. 位置编码
attention 本身没有顺序概念，要手动把「位置信息」加进输入（正弦编码 / 可学习位置）。

### 6. 完整结构
`Embedding + 位置编码 → N × [self-attention + FFN + 残差 + LayerNorm] → softmax`。

### 7. 代码
- MultiHeadAttention
- TransformerBlock
- 残差连接 + LayerNorm

### 8. 和 GPT 的关系
GPT = transformer 的 **decoder 半边**（只做 self-attention，自回归生成下一个词）。你现在写的这个，去掉编码器、加上预训练和规模，就是 GPT 的雏形。

## 关键可视化

- attention 权重热图（一个词在「看」哪些词）
- 生成文本的逐步变化

## 与上一版的关系

v11 的 attention 从「跨两个序列」变成 v12 的「序列内自我关注」，并叠加多头、位置编码、残差。至此，从 v1 线性回归到 transformer，完整走完了「LLM」的前半段旅程。
