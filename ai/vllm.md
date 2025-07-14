### PagedAttention
 - PagedAttention
 借鉴了操作系统中的虚拟内存分页机制，将 LLM 推理过程中生成的 Key-Value（KV）缓存分成固定大小的“块”（类似内存页），从而动态管理显存分配，避免传统方法中因连续内存分配导致的显存碎片化问题。
 - 张量并行
 - tensor_parallel_size 使用张量并行来运行模型，提高模型的处理吞吐量，分布式服务。
   需要num_attention_heads整除
 - data_parallel_size 通过复制模型到多个设备，并行处理不同输入数据，加速训练或推理。
 - 连续批处理（Continuous Batching）
 - CUDA Graphs
 - 模型量化（Quantization）
   AWQ（即激活值感知的权重量化，Activation-aware Weight Quantization)
   GPTQ（针对类GPT大型语言模型的量化方法）
   
 - 模型并行
前缀缓存（Prefix Caching）
推测解码（Speculative Decoding)
