### Attention Is All You Need
https://arxiv.org/pdf/1706.03762v7
# Attention、Self-Attention 与 Multi-Head Attention 详解

## 1. 注意力机制 (Attention)
### 数学表达 (Scaled Dot-Product Attention)
```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```
参数说明：
 - Q (Query)：当前要计算的目标（如解码器的某个词）。
 - K (Key)：输入序列的各个部分（如编码器的所有词）。
 - V (Value)：与 Key 对应的实际信息。
 - dₖ ：Key 的维度，用于缩放（防止 softmax 梯度消失）。

## 2. 自注意力 (Self-Attention)
特点：

Query、Key、Value都来自同一输入序列

能直接建模序列元素间的长距离依赖

支持并行计算

计算过程
线性变换：

math
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
计算注意力：

math
\text{Self-Attention}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
<a id="3-多头注意力-multi-head-attention"></a>

## 3. 多头注意力 (Multi-Head Attention)
结构：

math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W_O
其中：

math
\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)
优势：

多视角建模（不同头关注不同模式）

增强模型表达能力

<a id="对比总结"></a>

对比总结
机制	特点	主要用途
Attention	动态关注输入的不同部分（Q和K/V可不同源）	Seq2Seq、图像描述生成
Self-Attention	Q、K、V同源，捕捉序列内部关系	Transformer编码器/解码器
Multi-Head Attn	多组自注意力并行计算，提升模型表达能力	Transformer的核心组件
