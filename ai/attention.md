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
 - 多头注意力 是自注意力的扩展，通过 并行计算多组注意力，让模型从不同角度学习信息。
 - 在Transformer架构的LLM（Large Language Model）中，num_attention_heads（注意力头的数量）和 head_dim（每个注意力头的维度）是两个紧密相关的超参数，它们共同决定了模型隐藏层（hidden_size）的结构.
 - 模型的 隐藏维度（hidden_size） 是 num_attention_heads 和 head_dim 的乘积：
```
hidden_size = num_attention_heads × head_dim
```
 - 优势：
   - 多视角建模：不同头可以关注不同模式（如语法、语义、指代关系）。
   - 增强表达能力：比单头注意力更灵活。
<img width="1215" height="405" alt="image" src="https://github.com/user-attachments/assets/68a87c2e-6388-4e20-82a4-d48bb630d715" />


## 数学
 - 张量
   -  本质：张量是多维数组，可以看作一个统一的数据容器。
   -  与数学术语的关系：
   -  标量（0维张量）：单个数字，如 3.0。
   -  向量（1维张量）：一列数字，如 [1, 2, 3]。
   -  矩阵（2维张量）：二维数字表格，如 [[1, 2], [3, 4]]。
   -  高阶张量（n维）：三维及以上数组，如图像（高度×宽度×通道）、视频（时间×高度×宽度×通道）等。
 - 矩阵相乘: (猫和鱼在不同维度下的关注度的和, 两两文字间在不同维度下特征值的加权求和, 模型训练的目的是 反反复复在找猫和鱼在不同维度下的对应关系,并找到最佳的关系)
  <img width="1076" height="1343" alt="image" src="https://github.com/user-attachments/assets/f247bf2d-a33e-4aa7-a0cf-3c1bba4c8931" />

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

### 训练的真正目的
<img width="1204" height="711" alt="image" src="https://github.com/user-attachments/assets/e1167a35-af44-477f-8cc9-769e8a790add" />

### MHA、MQA、GQA
<img width="1737" height="1254" alt="image" src="https://github.com/user-attachments/assets/133d03e9-1a3f-4865-9a46-f96c048f0771" />
<img width="1654" height="926" alt="image" src="https://github.com/user-attachments/assets/aa3892aa-7cad-461f-a8a7-bcec7d9b06bb" />

### Sparse Attention (Google AI)
<img width="862" height="591" alt="image" src="https://github.com/user-attachments/assets/b3262a9f-bb03-4c63-a165-c8cfe502fe78" />
### Infini Attition (训练和生成都可用) (Google AI - Gemini 1.5)

#### Linear Attition (所有过去单词的每个特征值的加权和)
<img width="2167" height="1406" alt="image" src="https://github.com/user-attachments/assets/5d746a7b-3221-42b6-bf44-e4462283cc23" />
<img width="932" height="609" alt="image" src="https://github.com/user-attachments/assets/135fb44c-c663-45d9-b68c-3b155d69996f" />

### 支持超长上下文技术
<img width="942" height="605" alt="image" src="https://github.com/user-attachments/assets/888f4cd1-cf07-4fdc-9cdd-f0fca0e41a97" />

### 位置编码  Sinusoidal, RoPE , AliBi
   <img width="900" height="900" alt="image" src="https://github.com/user-attachments/assets/264743f4-ef86-47ea-a7ac-7160f921d730" />

 - Sinusoidal 正余弦位置编码 语言翻译
   
 - RoPE (Rotary Position Embedding ) (旋转位置编码) 影响Q,K 不影响V https://arxiv.org/pdf/2310.05209
   <img width="900" height="900" alt="image" src="https://github.com/user-attachments/assets/11cf77ed-f58e-444c-8b13-a4434804cff4" />

 - YaRN(Yet Another Rotary Notebook) 针对性的微调
  NTK-Aware / Dynamic-NTK (Code Llama/ Qwen 7B 使用)
  PI (Postion Interpolation)
  YaRN 基于 PI 和 NTK只上对于 RoPE的延展
 - ALiBi (线性偏置项位置编码)
   <img width="900" height="900" alt="image" src="https://github.com/user-attachments/assets/55b743e9-9448-402f-a5d7-4e4fb2d3ad19" />

   

