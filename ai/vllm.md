### 支持的Model
https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models
### 加速优化
 - PagedAttention
 借鉴了操作系统中的虚拟内存分页机制，将 LLM 推理过程中生成的 Key-Value（KV）缓存分成固定大小的“块”（类似内存页），从而动态管理显存分配，避免传统方法中因连续内存分配导致的显存碎片化问题。
   https://blog.vllm.ai/2023/06/20/vllm.html
 - 张量并行tensor_parallel_size 使用张量并行来运行模型，提高模型的处理吞吐量，分布式服务。
   需要num_attention_heads整除 https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json#L17
 - 前缀缓存（Prefix Caching） enable_prefix_caching=True 默认开启,（如共享相同提示词的多个生成任务）. 在聊天场景或多轮对话中，可减少 30%-70% 的计算量。
 - data_parallel_size 通过复制模型到多个设备，并行处理不同输入数据，加速训练或推理。
 - pipeline_parallel_size 流水线并行阶段数,将模型层（layers）拆分切分到多少个GPU上) 
 - Continuous Batching 连续批处理（Continuous Batching）
   一种高效批处理技术，旨在动态合并多个请求的执行过程，最大化 GPU 利用率，显著提升吞吐量（尤其是高并发场景）。其核心思想是打破传统静态批处理的限制，允许随时加入新请求，并灵活管理不同请求的生命周期。

| **特性**               | **静态批处理（Static Batching）**               | **连续批处理（Continuous Batching）**          |
|------------------------|-----------------------------------------------|---------------------------------------------|
| **请求调度-调度器与 GPU 计算解耦**   | 所有请求同时开始，同时结束       | 新请求可随时加入，完成请求立即释放资源     |
| **GPU利用率 (高~100%,提升5-10倍)**   | 低（受限于最慢的请求）      | 高（动态填充空闲计算单元）            |
| **适用场景**           | 批量生成（如离线任务）                         | 在线服务（如聊天、API 接口）                 |
| **延迟公平性**         | 所有请求等待最慢的完成                         | 先完成的请求先返回                           |
| **显存管理**           | 预分配固定显存，可能浪费                       | 动态分配/释放显存（如 PagedAttention）       |
| **吞吐量**             | 低（批处理大小固定）                           | 高（支持动态扩缩容）                         |
| **实现复杂度**         | 简单                                          | 高（需调度器、异步控制）                     |


 - CUDA Graphs
 - 模型量化（Quantization）[llm-compressor](http://github.com/vllm-project/llm-compressor)
   <img width="1478" height="1145" alt="image" src="https://github.com/user-attachments/assets/90602797-1371-46b6-a49a-8891a23dabf9" />
   AWQ（即激活值感知的权重量化，Activation-aware Weight Quantization)
      即支持 GPU 也支持 CPU
      使用AWQ量化的模型不会量化所有权重;而是量化对模型不重要的权重，以保持模型的有效性，
   GPTQ（针对类GPT大型语言模型的量化方法）
    (General post-training quantization)
    
    <img width="1347" height="792" alt="image" src="https://github.com/user-attachments/assets/31cb3a9d-4f58-4ef9-8952-a36343b6012a" />


 - 推测解码（Speculative Decoding)

# 参数详解

## 1. 核心参数 (vLLM 示例)

| 参数名                   | 作用                                                                 | 典型值   | 注意事项                                |
|--------------------------|---------------------------------------------------------------------|----------|----------------------------------------|
| `max_num_seqs`           | 单批次支持的最大请求数(批处理大小)                                   | 256      | 过大会增加显存压力                    |
| `max_num_batched_tokens` | 单批次允许的最大总token数(输入+生成)                                 | 2048     | 需根据模型和GPU显存调整               |
| `max_model_len`          | 单请求允许的最大上下文长度(输入+输出tokens)                          | 4096     | 超出会触发截断或错误                  |
| `batch_size`            | (部分框架)初始批处理大小                                            | 8        | 动态批处理下实际值可能更高            |
| `enable_prefix_caching`  | 是否启用前缀共享优化(默认`True`)                                     | True     | 对共享前缀的请求显著提升性能          |

## 2. 动态调度控制参数

| 参数名              | 作用                                   | 示例值       |
|---------------------|---------------------------------------|-------------|
| `scheduler_policy`  | 调度策略(如FIFO/优先级)               | "fifo"      |
| `preemption_mode`   | 是否允许抢占资源                      | "swap"      |
| `timeout_s`        | 请求最大等待时间(秒)                  | 30          |

## 3. 性能调优参数

| 参数名                      | 作用                                   | 调整建议                     |
|----------------------------|---------------------------------------|----------------------------|
| `block_size`               | KV缓存块大小(tokens/块)              | 16/32/128(需测最优值)      |
| `gpu_memory_utilization`   | GPU显存利用率目标(0~1)                | 0.9(接近上限但避免OOM)     |
| `swap_space`               | 允许KV缓存交换到CPU内存的大小(GiB)    | 4(长上下文时增加)          |
| `tensor_parallel_size`     | 张量并行 (需要模型的num_attention_heads整除)  |     多注意力头的并行计算    |
| `pipeline_parallel_size`   | 流水线并行阶段数|  将模型层（layers）拆分切分到多少个GPU上)   |
| `data_parallel_size`       | 数据批次 | 将数据批次（batch）拆分到多个GPU，每个GPU持有完整模型副本)   |

 总CPU需满足pipeline_parallel_size * tensor_parallel_size * data_parallel_size = 总GPU数
 
## 4. 请求级参数(生成时指定)

```python
output = engine.generate(
    prompt="Hello, how are you?",
    max_tokens=50,           # 最大生成tokens
    ignore_eos=False,        # 是否忽略EOS停止生成
    temperature=0.8,         # 影响采样随机性
    stream=True,             # 是否流式输出
)
```

## 5. 观察停止原因 output.finish_reason
finish_reason 字段解释了生成结束的原因，常见值包括：
"length"：达到 max_tokens 限制（可能被截断）。
"stop"：遇到停止词（如用户定义的终止字符串）。
"eos_token"：遇到模型的结束符（如 <|endoftext|>）。
None：生成尚未完成（流式输出时可能暂未结束）。
