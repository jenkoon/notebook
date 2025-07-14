### 支持的Model
https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models
### 加速优化
 - PagedAttention
 借鉴了操作系统中的虚拟内存分页机制，将 LLM 推理过程中生成的 Key-Value（KV）缓存分成固定大小的“块”（类似内存页），从而动态管理显存分配，避免传统方法中因连续内存分配导致的显存碎片化问题。
   https://blog.vllm.ai/2023/06/20/vllm.html
 - 张量并行 tensor_parallel_size 使用张量并行来运行模型，提高模型的处理吞吐量，分布式服务。
   需要num_attention_heads整除 https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json#L17
 - 前缀缓存（Prefix Caching） enable_prefix_caching=True 默认开启,（如共享相同提示词的多个生成任务）. 在聊天场景或多轮对话中，可减少 30%-70% 的计算量。
 - data_parallel_size 通过复制模型到多个设备，并行处理不同输入数据，加速训练或推理。
 - 连续批处理（Continuous Batching）
   
 - CUDA Graphs
 - 模型量化（Quantization）
   AWQ（即激活值感知的权重量化，Activation-aware Weight Quantization)
   GPTQ（针对类GPT大型语言模型的量化方法） 
 - 模型并行
 - 推测解码（Speculative Decoding)

### Attention Is All You Need
https://arxiv.org/pdf/1706.03762v7


 - 观察停止原因 output.finish_reason
finish_reason 字段解释了生成结束的原因，常见值包括：
"length"：达到 max_tokens 限制（可能被截断）。
"stop"：遇到停止词（如用户定义的终止字符串）。
"eos_token"：遇到模型的结束符（如 <|endoftext|>）。
None：生成尚未完成（流式输出时可能暂未结束）。
