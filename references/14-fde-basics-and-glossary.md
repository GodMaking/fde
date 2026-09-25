> 由 SKILL.md 路由加载：FDE 基础认知（是什么、有哪几类）与术语表——快速对齐概念、现场用词。

# FDE 基础认知与术语表

## 一、FDE 是什么

**FDE = Forward / Frontier Deployed Engineer，前沿部署工程师。**

定义：把前沿 AI 模型从「能跑」变成「能在线上稳定、高效、低成本地跑」的人，是 AI 模型能力与真实业务需求之间的桥梁。

四条核心职责：

1. **降低推理成本** —— GPU 利用率优化、量化、批处理
2. **提升推理性能** —— 延迟、吞吐、并发
3. **保障服务稳定性** —— 容灾、监控、弹性扩缩容
4. **适配最新模型** —— 新模型快速上线

**类比**：研究科学家造发动机、ML 工程师造车、FDE 修高速并管车队。

**FDE 不做什么**：不写论文；不懂底层的纯业务开发；不会开发的纯运维；只收框架、没有深度的工具收集癖。

## 二、FDE 的三种类型

| 类型 | 适用场景 | 客户是谁 | 交付什么 | 判断标准 |
|------|---------|---------|---------|---------|
| **Type A 推理基础设施型** | 推理引擎优化、GPU 集群与调度、推理平台建设 | 内部 Infra / Platform 团队 | 生产级推理服务，延迟 / 吞吐 / 成本达标 | JD 含 vLLM / TensorRT / GPU / CUDA；归属 Infra / Platform |
| **Type B 应用部署型** | AI 应用架构落地，把模型能力接进业务流程 | 内部 AI 应用团队、业务线 | RAG 管线、Agent 框架、模型评估体系 | JD 含 RAG / Agent / LangChain；归属 AI 应用团队 |
| **Type C 客户交付型（Palantir 模式）** | 客户现场部署、业务流程 AI 化改造 | 外部企业客户 | 可运行的系统 + 技术方案与咨询 + 业务改造结果 | JD 含 Customer / Client / Delivery；归属 GTM / Enterprise |

三类都要求技术 + 行业理解 + 沟通，差别在重心：Type A 要深度（GPU / 网络 / 编译器），Type B 要广度（全栈 AI 应用），Type C 要综合。

## 三、术语表

### 模型架构

| 术语 | 英文 | 一句话解释 | 现场怎么用 |
|------|------|-----------|-----------|
| Transformer | Transformer | 现代 LLM 的基础架构，核心是 Attention | 别说「神经网络模型」，说「底层架构是 Transformer」 |
| 注意力机制 | Self-Attention | 让每个 token 关注序列中其它 token 的机制 | 解释长文本更贵：计算量随长度平方增长 |
| KV 缓存 | KV Cache | 缓存 Decode 阶段已算好的 K / V 矩阵，避免重复计算 | 解释显存占用与并发上限的根因 |
| MHA | Multi-Head Attention | 每个 head 有独立的 Q / K / V | 传统做法，作对比基线 |
| GQA | Grouped Query Attention | 多个 query head 共享一组 K / V，减小 KV 缓存 | 说明「为什么能扛更高并发」 |
| MQA | Multi-Query Attention | 所有 query head 共享同一组 K / V | 压缩更极端，用于显存极敏感场景 |
| MLA | Multi-Latent Attention | 低秩压缩 KV 的 Attention 变体 | 讲 DeepSeek 系列时用到 |
| RoPE | Rotary Position Embedding | 通过旋转矩阵注入位置信息 | 客户问长上下文时涉及 |
| MoE | Mixture of Experts | 每层多个「专家」，Router 为每个 token 只激活部分 | 解释「为什么 1T 参数也跑得起」 |
| FFN | Feed-Forward Network | Transformer 中的前馈网络，占大多数参数 | 说明参数量分布与量化收益点 |
| SwiGLU | Swish-Gated Linear Unit | 现代模型使用的 FFN 激活函数 | 只在深挖结构时提 |
| RMSNorm | Root Mean Square Normalization | 只计算均方根的轻量 Normalization | 同上，结构细节 |

### 推理性能指标

| 术语 | 英文 | 一句话解释 | 现场怎么用 |
|------|------|-----------|-----------|
| TTFT | Time To First Token | 从发请求到看到第一个 token 的时间，越低越好 | 体验类指标，客户可直接感知 |
| TPOT | Time Per Output Token | 每生成一个 token 所需时间，越低越好 | 决定「打字速度」 |
| QPS | Queries Per Second | 每秒处理的请求数 | 谈容量规划 |
| 吞吐量 | Throughput | 每秒生成的 token 数（tokens/sec/GPU） | 谈成本效率，要带 GPU 单位 |
| 并发 | Concurrency | 同时服务的用户 / 请求数 | 谈「能同时服务多少人」 |
| 百分位延迟 | P50 / P99 | P50 是中位数，P99 表示 99% 的请求不超过该值 | 谈稳定性用 P99，别只报平均值 |

### 推理引擎与工具

| 术语 | 英文 | 一句话解释 | 现场怎么用 |
|------|------|-----------|-----------|
| vLLM | vLLM | 最流行的开源 LLM 推理引擎，核心是 PagedAttention | 多数场景的默认方案 |
| 分页注意力 | PagedAttention | 用分页方式管理 KV 缓存，减少显存碎片 | 解释 vLLM 为什么省显存 |
| 连续批处理 | Continuous Batching | 动态调度 batch，不等最长请求跑完 | 解释吞吐提升从哪来 |
| 在途批处理 | In-flight Batching | 更细粒度的 batch 调度 | TensorRT-LLM 侧的说法 |
| TGI | Text Generation Inference | Hugging Face 官方推理框架 | Hugging Face 生态配套时提 |
| TensorRT-LLM | TensorRT-LLM | NVIDIA 官方推理编译器，追求极致性能 | 整机 NVIDIA、追求极致性能时提 |
| SGLang | SGLang | 擅长结构化生成与前缀缓存的推理框架 | 结构化输出、前缀复用场景 |
| LangGraph | LangGraph | Agent 编排框架 | 谈 Agent 工作流编排 |
| Ollama | Ollama | 本地运行 LLM 的简易工具 | 谈本地试跑与私有化验证 |
| 投机解码 | Speculative Decoding | 用小模型预测、大模型验证来加速生成 | 谈「还能不能再提速」 |
| Flash Attention | Flash Attention | 减少显存访问的高效 Attention 算法 | 谈长上下文与显存优化 |

### 并行与分布式

| 术语 | 英文 | 一句话解释 | 现场怎么用 |
|------|------|-----------|-----------|
| 张量并行 | TP / Tensor Parallel | 把层内矩阵切到多张 GPU | 单机多卡装不下一个模型时 |
| 流水线并行 | PP / Pipeline Parallel | 把模型按层分配到不同 GPU | 跨机部署大模型时 |
| 数据并行 | DP / Data Parallel | 多个模型副本并行处理不同请求 | 扩容、提吞吐时 |
| 专家并行 | EP / Expert Parallel | MoE 中不同 Expert 分布在不同 GPU | MoE 大模型部署必提 |
| AllReduce | AllReduce | 多 GPU 同步数据的集合通信操作 | 谈通信瓶颈 |
| NCCL | NVIDIA Collective Communications Library | NVIDIA 官方 GPU 通信库 | 谈多卡多机通信 |
| MIG | Multi-Instance GPU | 把一张 GPU 切成多个独立实例 | 谈多租户与资源隔离 |

### 量化

| 术语 | 英文 | 一句话解释 | 现场怎么用 |
|------|------|-----------|-----------|
| 量化 | Quantization | 用更低精度存储模型以减少显存并加速 | 客户问「怎么降本」的第一答案 |
| 训练后量化 | PTQ / Post-Training Quantization | 训练后量化，不需重新训练 | 交付快、成本低，优先推荐 |
| 量化感知训练 | QAT / Quantization-Aware Training | 量化感知训练，精度更高但需重训 | 精度要求极高时才提 |
| 标准浮点 | FP32 | 32 位浮点，每参数 4 字节 | 作为精度基线 |
| 半精度 | FP16 | 16 位浮点，每参数 2 字节 | 常规部署精度 |
| BF16 | Brain Float 16 | 改进的半精度，数值范围更适合训练 | 说明常用精度 |
| FP8 | 8-bit Floating Point | 8 位浮点，新卡原生支持 | 谈新硬件的收益 |
| INT8 | 8-bit Integer | 8 位整数，每参数 1 字节 | 通用量化档位 |
| INT4 | 4-bit Integer | 4 位整数，每参数 0.5 字节 | 显存最紧张时的档位 |
| AWQ | Activation-aware Weight Quantization | 按激活重要性保护关键权重的量化 | 4bit 部署常用方案 |
| GPTQ | Generative Pre-trained Quantization | 逐层贪心量化，用 Hessian 加权 | 另一种常用 4bit 方案 |
| SmoothQuant | SmoothQuant | 平滑激活与权重分布，便于同时量化 | 激活量化难时的方案 |

### AI 工程

| 术语 | 英文 | 一句话解释 | 现场怎么用 |
|------|------|-----------|-----------|
| 检索增强生成 | RAG | 先从知识库检索，再让模型基于检索结果生成 | 客户说「用我们自己的资料问答」时 |
| 智能体 | Agent | 能调用工具、自主规划执行任务的系统 | 客户说「让 AI 自动干活」时 |
| 提示词 | Prompt | 给模型的输入指令 | Prompt Engineering = 提示词工程 |
| 嵌入向量 | Embedding | 把文本转成数值向量 | 解释检索与相似度怎么实现 |
| 向量数据库 | Vector DB | 存储和检索 Embedding 的数据库 | RAG 方案里的必备组件 |

### GPU 硬件

| 术语 | 英文 | 一句话解释 | 现场怎么用 |
|------|------|-----------|-----------|
| 高带宽显存 | HBM / High Bandwidth Memory | GPU 的高带宽显存 | 谈「显存够不够装下模型」 |
| 流式多处理器 | SM / Streaming Multiprocessor | GPU 的基本执行单元 | 谈利用率与算力 |
| 张量核心 | Tensor Core | 专为矩阵乘法设计的硬件单元 | 谈算力利用率 |
| CUDA 核心 | CUDA Core | 标量计算单元 | 与 Tensor Core 对照 |
| NVLink | NVLink | NVIDIA GPU 之间的高速互联 | 谈多卡通信带宽 |
| 线程束 | Warp | GPU 中 32 个线程为一组同时执行 | 底层性能调优时 |

### 部署与运维

| 术语 | 英文 | 一句话解释 | 现场怎么用 |
|------|------|-----------|-----------|
| 服务级别目标 | SLO / Service Level Objective | 内部设定的服务质量目标 | 与客户量化约定稳定标准 |
| 服务级别协议 | SLA / Service Level Agreement | 对外承诺的服务级别 | 合同层面的承诺 |
| 站点可靠性工程 | SRE / Site Reliability Engineering | 用工程手段保障可靠性 | 谈运维保障体系 |
| 水平扩缩容 | HPA / Horizontal Pod Autoscaler | K8s 按负载自动增减副本 | 谈弹性扩缩容 |
| Pod | Pod | K8s 最小部署单元 | 谈部署形态 |
| 容器编排 | K8s / Kubernetes | 容器编排平台 | 谈交付与部署环境 |
| A/B 测试 | A/B Testing | 两个版本对比测试 | 谈新模型 / 新版本效果验证 |
| 灰度发布 | Canary | 逐步放量上线 | 谈上线风险控制 |
| 服务端推送 | SSE / Server-Sent Events | 服务端流式推送响应 | 谈「打字机效果」怎么实现 |

### 训练与其它

| 术语 | 英文 | 一句话解释 | 现场怎么用 |
|------|------|-----------|-----------|
| 监督微调 | SFT / Supervised Fine-Tuning | 用标注数据监督微调 | 客户提「定制模型」时的第一步 |
| 人类反馈强化学习 | RLHF | 用人类反馈做强化学习对齐 | 谈对齐与偏好优化 |
| 直接偏好优化 | DPO / Direct Preference Optimization | 比 RLHF 更简的偏好对齐方法 | 偏好优化提成本时的替代方案 |
| 当前最优 | SOTA / State Of The Art | 当前最优水平 | 谨慎使用，必须给具体指标 |

## 四、最容易混淆的几组概念

| A vs B | 差在哪 | 跟客户说的时候用哪个 |
|--------|-------|-------------------|
| FDE vs ML 工程师 | ML 工程师产出训练好的模型；FDE 让模型在生产环境稳定高效低成本地跑 | 对客户讲「我把模型变成可用服务」，不要自称算法工程师 |
| FDE vs AI 应用工程师 | 应用工程师交付 RAG / Agent 应用；FDE 交付应用背后的推理服务与平台能力 | 客户要「能用的产品」→ 应用工程师；要「跑得稳、跑得便宜」→ FDE |
| 推理优化 vs 训练优化 | 训练优化关心收敛与训练成本；推理优化关心延迟、吞吐、并发与单位成本 | 现场默认讲推理优化，除非客户在自训模型 |
| 延迟 vs 吞吐 | 延迟是「一个请求多快」，吞吐是「单位时间能处理多少」；二者常互相拉扯 | 面向体验谈延迟，面向成本谈吞吐 |
| SLO vs SLA | SLO 是内部目标，SLA 是对外合同承诺 | 内部对齐用 SLO，写进合同用 SLA，别把内部目标当承诺说出口 |
| TP vs PP | TP 是层内把矩阵切开，PP 是按层切分到不同 GPU | 单机多卡讲 TP，跨机讲 PP，实际常组合使用 |
| 灰度发布 vs A/B 测试 | 灰度是逐步放量控风险，A/B 是按流量分流比效果 | 控风险讲 Canary，比效果讲 A/B Testing |
| PTQ vs QAT | PTQ 训练后量化、交付快；QAT 需重训、精度更高 | 先给客户 PTQ 方案，精度不达标再谈 QAT |
