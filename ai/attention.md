## Attention Is All You Need
https://arxiv.org/pdf/1706.03762v7
# Attention、Self-Attention 与 Multi-Head Attention

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
自注意力 是注意力机制的一种特殊形式，它的 Query、Key、Value 都来自同一输入序列，让序列内部的元素互相计算相关性。

特点:
 - 捕捉长距离依赖：无论词之间的距离多远，自注意力都能直接建模它们的关系（不像 RNN 依赖逐步传递）。
 - 并行计算：所有位置的注意力权重可以同时计算，效率高于 RNN。

## 3. 多头注意力 (Multi-Head Attention)
多头注意力 是自注意力的扩展，通过 并行计算多组注意力，让模型从不同角度学习信息。
优势：
 - 多视角建模：不同头可以关注不同模式（如语法、语义、指代关系）。
 - 增强表达能力：比单头注意力更灵活。
<img width="1215" height="405" alt="image" src="https://github.com/user-attachments/assets/68a87c2e-6388-4e20-82a4-d48bb630d715" />



## 数学
 - linear
   在 Transformer 模型中，Linear（线性层）指的是一个全连接神经网络层（Fully Connected Layer），其数学本质是 线性变换（即矩阵乘法 + 偏置）。它的核心作用是对输入数据进行 维度变换 或 特征空间映射。
   <img width="1079" height="1421" alt="image" src="https://github.com/user-attachments/assets/db9e5582-c471-43d7-bb5d-0e753d59e239" />

 - Layer Normalzation
   nn.LayerNorm(bias=True)
 - Softmax
   torch.softmax()
## QKV
<img width="1030" height="1261" alt="image" src="https://github.com/user-attachments/assets/42a0820e-06ba-4945-8552-2429ea553603" />
<img width="1043" height="1332" alt="image" src="https://github.com/user-attachments/assets/7e9cc480-eb81-4428-a055-b2e620bf4ab5" />
<img width="1046" height="1341" alt="image" src="https://github.com/user-attachments/assets/78cee859-9199-4374-9212-2e5842f9c367" />
<img width="2141" height="1160" alt="image" src="https://github.com/user-attachments/assets/ed22c2a4-6d1e-4287-9d1f-4f6d6d0eef34" />
<img width="2120" height="1410" alt="image" src="https://github.com/user-attachments/assets/22446a36-b683-4893-863c-7d51acafd673" />


