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
   一种高效批处理技术，旨在动态合并多个请求的执行过程，最大化 GPU 利用率，显著提升吞吐量（尤其是高并发场景）。其核心思想是打破传统静态批处理的限制，允许随时加入新请求，并灵活管理不同请求的生命周期。

| **特性**               | **静态批处理（Static Batching）**               | **连续批处理（Continuous Batching）**          |
|------------------------|-----------------------------------------------|---------------------------------------------|
| **请求调度**           | 所有请求同时开始，同时结束                     | 新请求可随时加入，完成请求立即释放资源       |
| **GPU 利用率**         | 低（受限于最慢的请求）                         | 高（动态填充空闲计算单元）                   |
| **适用场景**           | 批量生成（如离线任务）                         | 在线服务（如聊天、API 接口）                 |
| **延迟公平性**         | 所有请求等待最慢的完成                         | 先完成的请求先返回                           |
| **显存管理**           | 预分配固定显存，可能浪费                       | 动态分配/释放显存（如 PagedAttention）       |
| **吞吐量**             | 低（批处理大小固定）                           | 高（支持动态扩缩容）                         |
| **实现复杂度**         | 简单                                          | 高（需调度器、异步控制）                     |


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
