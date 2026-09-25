> 由 SKILL.md 路由加载：推理引擎选型、量化、调优、POC 评测验收。
# 推理引擎与推理优化 · 现场干活手册

> 适用对象：在客户现场把 LLM 跑起来、调快、调便宜、验收达标的 FDE。
> 默认读者已理解 Transformer 自回归生成、KV Cache、GPU 显存层级。删去原理推导，只留可操作结论、数字与命令。

---

## 1. 三大引擎选型决策规则

### 1.1 对照表（带数字）

| 维度 | vLLM | SGLang | TensorRT-LLM (TRT-LLM) |
|------|------|--------|------------------------|
| 核心创新 | PagedAttention | RadixAttention + FSM 结构化生成 | 编译时 kernel 融合 + 精度校准 |
| 吞吐量 | 高 | 高（接近 vLLM） | 最高，约 vLLM 的 1.2–2x（同硬件） |
| 延迟 | 低 | 低 | 最低 |
| 模型支持 | 最广（50+ 架构） | 较广 | 受限（仅 NVIDIA 验证列表，新增需写适配） |
| 硬件 | NVIDIA/AMD/Intel | 仅 NVIDIA | 仅 NVIDIA |
| 量化 | AWQ/GPTQ/FP8/INT8 | AWQ/GPTQ/FP8/INT8 | INT8/FP8 自动校准 |
| 易用性 | 极好（`pip install`，秒级启动） | 好 | 中（需 build，10–30 分钟） |
| 结构化输出 | 有限（Outlines 集成） | 原生 FSM（JSON/Regex/CFG 100% 合规） | 无原生 |
| Agent 前缀复用 | Prefix Caching（block 级精确匹配） | RadixAttention（前缀树，2–5x） | 一般 |
| 生产成熟度 | 生产就绪 | 快速迭代中 | 生产就绪 |

### 1.2 决策树（直接用）

1. **是否只用 NVIDIA GPU？**
   - 否 / 混合 → **vLLM**（多后端支持）。
   - 是 → 进入 2。
2. **是否以 Agent / 结构化输出（Function Calling、JSON、代码）为主？**
   - 是 → **SGLang**（RadixAttention 复用 system prompt + FSM 强制格式，消除"生成→解析→重试"）。
   - 否 → 进入 3。
3. **是否需要极致性能（延迟 < 10ms 或吞吐 > 100K tok/s），且模型在 TRT-LLM 支持列表、能接受 build 时间和低灵活性？**
   - 是 → **TRT-LLM**（通常比 vLLM 快 20%–100%，A100 上 kernel fusion 额外 1.3–1.5x）。
   - 否 → **vLLM**（生态最稳、迭代最快）。

**经验法则**：
- 通用聊天/补全/多模型多租户 → vLLM。
- 大量相同 system prompt + few-shot 的多轮对话、工具调用 → 优先 SGLang；命中率 < 30% 则退化。
- 单一模型大规模固定部署、实时翻译/对话机器人 → TRT-LLM + Triton。
- 不确定 → 先 vLLM，后续按需求评估 SGLang；混合架构：`通用请求→vLLM集群，Agent/结构化→SGLang集群`。
- TRT-LLM 的 In-flight Batching（token 级）比 vLLM Continuous Batching（序列级）batch 利用率高约 5–10%，但调度思想一致。

---

## 2. 可直接抄的启动命令与关键参数

### 2.1 vLLM 生产启动模板

```bash
vllm serve meta-llama/Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \        # >30B 设 2-8；小模型保持 1
  --max-num-seqs 256 \              # A100-80G 可 256-512；显存不足降
  --gpu-memory-utilization 0.9 \    # 独占 GPU 可 0.95；多进程共享需降
  --swap-space 16 \                 # 突发流量 16-32GB，但延迟高不建议依赖
  --enable-prefix-caching \         # 前缀命中高必开；否则有 2-5% hash 开销
  --max-model-len 8192 \            # 按业务设；过大会减少可并发请求数
  --quantization fp8 \              # H100 上 1.5-2x；A100 用 awq/gptq
  --max-num-batched-tokens 8192 \   # 建议 = max_model_len × max_num_seqs
  --disable-log-requests            # 高 QPS 减 5-10% I/O 开销
```

OpenAI 兼容调用：base_url 指向 `http://host:8000/v1`，无需改业务代码；支持 `/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`。不支持 function calling 原生（需 SGLang）。

### 2.2 SGLang 启动模板

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3-8B-Instruct \
  --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.85 \      # 类比 gpu_memory_utilization
  --max-running-requests 256        # 类比 max-num-seqs
```

结构化生成（Python）：用 `regex=` 或 `build_regex_from_schema(schema)` 强制 JSON 合规，开销 < 5%，换来 100% 格式正确率。

### 2.3 TRT-LLM 构建 + 部署

```bash
python build.py \
  --model_dir /path/to/model \
  --dtype bfloat16 \                # A100/H100 用 bf16
  --use_inflight_batching \         # 生产必开
  --tp_size 1 --world_size 1 --pp_size 1 \  # 70B: tp 4-8
  --max_batch_size 256 \
  --max_input_len 4096 --max_output_len 2048 \
  --remove_input_padding \          # 去 padding，+10-20% 吞吐
  --fp8                             # H100 必开；A100 不支持
```

生产推荐 TRT-LLM Backend 接入 **Triton Inference Server**（gRPC/HTTP/Streaming、动态 batch、多模型、Prometheus）。

### 2.4 关键参数调优方向（含公式）

- **`gpu_memory_utilization`**：KV Cache 可用显存 = `GPU总显存 × 0.9 − 权重显存 − 激活显存`。
  例（Llama-3-8B, A100-80G）：权重 16GB，可用 72GB，KV 预算 ≈ 52GB；每 token KV = `layers×2×hidden×2B` = 32×2×4096×2 ≈ 512KB → 约可存 10 万 token 的 KV。
- **`max_num_seqs`**：过小（16）→ GPU 空转，吞吐低，延迟低；适中（128–256）→ 平衡；过大（512+）→ 吞吐最高但 TTFT/TPOT 升，仅离线批推。
- **`max_model_len`**：过大 → KV 预算被占满，可并发数骤降；按业务 P99 上下文长度设，别拍脑袋拉满。
- **OOM 处理**：先降 `gpu_memory_utilization` 再降 `max-num-seqs`。
- **启用 prefix caching**：仅当请求有公共前缀（system prompt / few-shot / RAG 模板）时开，否则白付 2–5% 开销。

---

## 3. 量化方案对照与决策

### 3.1 显存/速度基准（以 70B 为例，A100-80G）

| 精度 | 单权重显存 | 加速比(decode) | 精度损失 | 硬件要求 |
|------|-----------|---------------|---------|---------|
| FP16 | 140 GB（需 2×A100） | 1.0x | 0 | 通用 |
| INT8 | 70 GB（1×A100） | 1.5–2x | MMLU <0.5% | 通用 |
| INT4 (AWQ/GPTQ) | 35 GB（1×A100） | 2–3x | MMLU -1~3% | 通用 |
| FP8 E4M3 | 70 GB | 1.5–2x（H100 达 1.8x） | <0.5% | **H100+** |

> 速度来源：A100 上 INT8 GEMM 624 TFLOPS vs FP16 312（2x），H100 上 FP8 1979 TFLOPS vs FP16 989（≈2x）。decode 是 memory-bound，权重体积减半 ≈ 延迟减半。

### 3.2 方案精度损失对照（Llama 级，相对 FP16 的 MMLU 下降）

| 方案 | Llama 7B | Llama 70B | 校准数据 | 量化耗时 |
|------|---------|----------|---------|---------|
| SmoothQuant INT8 | -0.3% | -0.3% | 128 | ~1 min |
| AWQ INT8 | -0.2% | -0.2% | 128 | ~1 min |
| AWQ INT4 (g=128) | -1.5% | -1.3% | 128 | ~2 min |
| GPTQ INT4 (g=128) | -2.8% | -1.9% | 128 | ~30 min |
| GPTQ INT4 (g=64) | -2.1% | -1.5% | 128 | ~60 min |
| FP8 E4M3 | -0.5% | -0.6% | 0 | 即时 |

### 3.3 决策规则

1. **硬件 H100** → 首选 **FP8 E4M3**（精度损失 <0.5%，速度最快，无需 dequantize）。权重 E4M3，激活 E5M2。
2. **硬件 A100 或更低** → 先看是否 INT8 够用：
   - 极致推理速度且需 INT8 → **SmoothQuant**（推理全程无 dequantize，更快）。
   - 通用 → **AWQ INT8**（精度最高）。
3. **必须 INT4（显存不够）**：
   - 要快（分钟级）→ **AWQ INT4**（g=128 是 sweet spot；64 精度略升但存储翻倍）。
   - 要极限压缩 → GPTQ INT4（逐层 Hessian，慢但压缩率略高）。
   - **首选 AWQ**：速度快、精度好、性价比高。
4. **FP8 格式选择**：E4M3（范围 ±448，精度高）用于权重/KV Cache；E5M2（范围 ±32768）用于激活（outlier 多）。

### 3.4 什么时候**不该**量化

- 数学推理、代码生成等对精度极度敏感的任务（INT4 MMLU 可掉 2–5%）。
- 模型已很小（<7B），量化收益有限。
- 显存充足且吞吐满足 SLA（没有瓶颈就别引入风险）。
- 后续要做 FP16 LoRA / 微调（adapter 通常用 FP16）。
- MoE 模型：expert 可 INT4，但 **router/gating 权重必须保持 FP16/INT8**（对精度极敏感）。

### 3.5 避坑清单

- INT4 量化后务必用 **GSM8K + 代码生成** 验证，别只看 MMLU。
- 校准数据 ≥ 200 条且**覆盖业务真实分布**（不能只用 Wikipedia），否则验证过拟合。
- per-group INT4 的 group_size = 128 最优；量化后首次推理慢（dequantize 预热），多 warmup 几次。
- 量化后跑关键 benchmark 确认损失 < 2%，并验证长上下文（量化对长序列更敏感）。

---

## 4. KV Cache 量化（显存瓶颈优化）

### 4.1 占比规律（Llama-3-70B）

- batch=1, seq=2048：KV 很小（≈0.33GB），量化收益不大。
- batch=64, seq=32768：KV ≈ 164GB，占显存 **54%**，是瓶颈根因。
- 结论：**单请求短序列先量化权重；多请求长序列先量化 KV Cache。**

### 4.2 方案对照

| 方案 | 压缩比 | MMLU 损失 | 实现难度 | GPU | 成熟度 |
|------|--------|-----------|---------|-----|--------|
| FP16（基准） | 1x | 0 | - | 任意 | - |
| **INT8** | 2x | <0.3% | 低 | 任意 | **高，推荐** |
| FP8 E4M3 | 2x | <0.2% | 低 | H100+ | 中-高 |
| K-INT4/V-INT2 (KIVI) | ~3x | 0.5–1.5% | 中 | 任意 | 中 |
| 全 INT4 | 2x | 1–2% | 低 | 任意 | 中（需验证） |
| PQ | 4–10x | 1–3% | 高 | 任意 | 低（实验性） |

### 4.3 启用方式（vLLM）

```python
LLM(model="...-AWQ-INT4", quantization="awq",
    kv_cache_dtype="int8",        # 或 "fp8"（H100）
    gpu_memory_utilization=0.95,  # 量化后可提到 0.95
    max_model_len=16384)
```
注意：`kv_cache_dtype` 需**重启引擎**生效，不支持运行时切换；量化后 prefix cache 也须存为量化格式。

### 4.4 坑

- INT8 KV 量化靠 **fused dequantize kernel**（开销 <5%）才净赚 20–40% decode 加速；差的独立 dequantize 实现可能反慢。
- FP8 KV（H100）无需 dequantize，天然更快（decode 延迟降 30–50%）。
- 质量敏感场景（数学/代码/多轮对话）KV 至少 INT8，不用 INT4。
- INT4 KV 仅显存极度紧张且容忍质量损失时用；70B+ 超长上下文可能轻微退化。

---

## 5. 投机解码（Speculative Decoding）

### 5.1 适用条件

- 只加速 **decode 阶段**（Prefill 不受益）。
- draft 与 target 接受率 > 50% 才有意义；**数学无损**（拒绝采样保证输出分布与直接跑大模型一致）。
- 理论加速 ≈ `1 + γ × 接受率`（γ = 每次 draft token 数，通常 3–6）。
- 实际加速（含 draft 开销）：代码补全 ~2.5x，翻译 ~2.2x，通用 ~2.0x，创意写作仅 ~1.5x。

### 5.2 接受率规律

- 高（70–85%）：代码补全、翻译、摘要（确定性高、模式重复）。
- 低（30–55%）：创意写作、开放问答、数学推理（每步强依赖前一步）。
- draft 与 target 差距越小、同源架构 → 接受率越高。**优先用 target 的同源量化版做 draft**。

### 5.3 启用（vLLM）

```python
LLM(model="Llama-2-70b",
    speculative_model="Llama-2-7b",  # draft
    num_speculative_tokens=4,        # γ，从 4 起调
    use_v2_block_manager=True)
```

### 5.4 调优与代价

- γ 从 4 起：接受率 >60% → 加到 5–6；<40% → 换 draft 或降到 2–3。
- 显存开销：draft 7B 约 14GB(FP16)/7GB(INT8)；70B+7B 组合约 154GB（2×H100 80G 刚好）。
- 反而不划算：接受率 <30%、输出 <20 token（overhead 占比大）、单卡无余量放 draft（改用 Medusa/EAGLE，单模型无额外显存）。
- 变体：Medusa（多头扩展，接受率 60–75%，加速 2–3x，需训练 K 个 head，~1–2 GPU 天）；EAGLE-3（特征预测层，2–6x，vLLM/SGLang 原生）。
- 兼容：Continuous Batching、PagedAttention、FP8/INT8 量化均可叠加；与多 LoRA（每请求不同）冲突。

---

## 6. Prefill / Decode 两阶段与 PD 分离

### 6.1 两阶段特征

| 阶段 | 计算性质 | 决定指标 | 瓶颈 |
|------|---------|---------|------|
| Prefill | 计算密集（compute-bound），并行处理整个 prompt | **TTFT** | GPU 算力 |
| Decode | 访存密集（memory-bound），逐 token 自回归 | **TPOT** | HBM 带宽 |

- TTFT ≈ `input_tokens × prefill_time_per_token`；TPOT 随 batch 增大而升高。
- 量化收益不对称：Prefill（compute-bound）加速比 <2x；Decode（memory-bound）权重减半 ≈ 延迟减半。

### 6.2 PD 分离（Prefill-Decode Disaggregation）

- 思路：Prefill 节点（compute-bound）与 Decode 节点（memory-bound）分开调度部署。
- 适用条件：**大流量服务（1000+ QPS）**，且两类负载特性差异显著、需独立扩缩容。
- 收益：吞吐 +2x，延迟 -40%（参考 DistServe / MoonCake 架构）。
- 代价：跨节点传输 KV Cache 的网络开销；架构复杂度高（需 KV 传输层 + 调度器）；流量不够大时收益覆盖不了复杂度。**中小流量先别上**。

---

## 7. 三大运行时优化（原理要点 + 启用条件）

### 7.1 Continuous Batching（连续批处理）

- 原理：不等整个 batch 完成，序列/ token 完成即回收 slot 并填入新请求；decode 请求优先（已投入 KV，避免浪费）。
- 收益：吞吐 ~4x（Llama-2-7B 实测 10→40 req/s），GPU 利用率 55%→85%，P99 降 3–5x。
- 启用：vLLM/SGLang/TRT-LLM 默认开启，无需额外配置。

### 7.2 Prefix Caching

- 原理：对 KV block 算 hash / 前缀树匹配，相同前缀（system prompt、few-shot、RAG 模板）的 KV 直接复用，跳过 prefill。
- 启用：vLLM `--enable-prefix-caching`；SGLang 默认 RadixAttention。
- 收益：前缀命中率 80% 时 TTFT 降 60–80%。
- 条件：**请求有公共前缀才开**；否则 2–5% hash 计算白付。

### 7.3 Chunked Prefill

- 原理：过长 prompt 拆成 chunk，避免单次 prefill 长时间霸占 GPU、饿死 decode。
- 启用：vLLM 由调度器在 decode 不满时机会性插入，配合 `max-num-batched-tokens` 控制单批 token 上限。
- 条件：长 prompt + 高并发、TTFT 排队严重时启用；本质是 prefill/decode 调度比例的平衡手段。

---

## 8. 性能调优标准动作顺序

### 8.1 关键指标关系

- **TTFT**：受 prefill 计算量 + 排队请求数影响（首 token 感知速度）。
- **TPOT**：受 decode 阶段 HBM 带宽 + batch size 影响（生成流畅度）。
- **吞吐**：增大 batch（Continuous Batching / PD 分离）↑，但与延迟冲突。
- **成本**：量化（FP8/INT4）+ 动态扩缩容 + 排队策略 ↓。
- 三角不可同时最优：提吞吐通常牺牲延迟，降成本通常影响两者。

### 8.2 调优顺序（先测后调，量化判断）

1. **基线测量**（先用 vLLM benchmark / k6）：固定矩阵测 TTFT(P50/P99)、TPOT、Decode 吞吐、最大并发。
   - Prefill 矩阵：input tokens = [100,500,1000,2000,4000,8000]，每长度 20 次取 P50/P99。
   - Decode 矩阵：batch = [1,4,8,16,32]，output = [100,200,500]。
2. **GPU 独占**（确认无其他进程抢卡）→ 否则数据全错。
3. **先解决 OOM / 排队**：降 `gpu_memory_utilization` 或 `max-num-seqs`；看 `vllm:num_requests_waiting` 判断 prefill 排队。
4. **提吞吐**：调 `max-num-seqs` 到 128–256（A100），开 FP8/INT8 量化（H100 上 +1.5–2x）。
5. **降延迟**：高 TTFT → chunked prefill / 降排队；高 TPOT → 降 `max-num-seqs` / 检查 GPU 利用率；考虑投机解码或 PD 分离。
6. **复用前缀**：有公共前缀开 prefix caching（命中率目标 >30% 才划算）。
7. **饱和后扩展**：单实例 GPU 利用率饱和 → 多实例 + Load Balancer，而非无限加参数。
8. **验收**：调好了的标准 = 在业务 SLA（如 P99 TTFT < 1s、TPOT < 30ms）下，GPU 利用率稳定 >70% 且 `gpu_cache_usage_perc` < 90%。

### 8.3 必监控指标（Prometheus `/metrics`）

- `vllm:num_requests_running` / `num_requests_waiting`、`gpu_cache_usage_perc`、`time_to_first_token_seconds`、`time_per_output_token_seconds`、`e2e_request_latency_seconds`。
- 告警：KV Cache 使用率 >90% 持续 5min → 扩容；P99 TTFT >1s → 查排队；GPU 利用率 <50% → batch 过小或加载问题。

### 8.4 常见故障排查

| 现象 | 原因 | 动作 |
|------|------|------|
| OOM | KV 超出显存 | 降 `gpu_memory_utilization` / `max-num-seqs` |
| 高 TTFT | Prefill 排队 | 查 waiting；开 chunked prefill |
| 高 TPOT | decode batch 过大 | 降 `max-num-seqs`；查 GPU 利用率 |
| 请求超时 | max_model_len 过小 | 调大 `max-model-len` |
| 显存泄漏 | 旧版本 bug | 升稳定版；查 swap |

---

## 9. 给客户做 POC 的评测 / 验收流程

### 9.1 四步法（总工期约 5–8 天）

1. **信息收集（0.5 天）**：读论文/公告、GitHub 讨论、benchmark 榜，确认核心 claim。
2. **环境搭建（1–2 天）**：装框架、下模型、跑通最小推理。
3. **基准测试（1–2 天）**：标准精度 + 性能指标 + 兼容性。
4. **业务验证（2–3 天）**：真实业务数据 A/B、成本收益分析，出报告。

### 9.2 看哪些指标（维度）

- **精度**：MMLU/CMMLU（通用）、GSM8K/BBH（推理）、HumanEval/MBPP（代码）、C-Eval/CLUE（中文）、TruthfulQA（安全）。新方案不低于现有方案。
- **性能**：TTFT(P50/P99)、TPOT、Prefill Latency、Decode 吞吐、最大并发（逐步加压至延迟超标）。
- **兼容性**：框架支持（vLLM/TGI/TRT-LLM/SGLang）、OpenAI API 兼容、量化支持（AWQ/GPTQ）、硬件（如 FP8 需 H100+）。
- **成本**：单 token GPU 成本 = GPU 时费 ÷ 吞吐；显存需求、GPU 张数、同 QPS 下成本差。
- **风险**：许可证（商用限制）、社区活跃度、长期维护、安全合规。

### 9.3 怎么算达标（量化阈值，按场景定 SLA）

- 精度：量化后关键 benchmark 损失 **< 2%**（FP8 <0.5%、INT8 <0.5%、INT4 接受 1–3% 但需业务验证）。
- 延迟：P99 TTFT < SLA（如 1s），TPOT < 30ms 为流畅基准；长上下文（>8K）允许放宽。
- 吞吐：在目标 QPS 下 GPU 利用率 >70%、KV Cache 使用率 <90%、错误率（OOM/超时）<0.1%。
- 成本：达标前提下单 token 成本 ≤ 现方案（或客户预算阈值）。
- 稳定性：10 QPS 持续 24h 无显存泄漏、延迟漂移。

### 9.4 验收报告必备（量化，不写"感觉快了"）

1. 结论摘要：推荐 / 有条件推荐 / 不推荐。
2. 精度对比表（现有 vs 新，附变化 pp）。
3. 性能对比表（TTFT/TPOT/吞吐，附变化 %）。
4. 成本表（单 token 成本、GPU 张数变化）。
5. 兼容性清单（勾选框架/API/量化/硬件）。
6. 风险（许可证/维护/精度）。
7. 适用与不适用场景。
8. 后续计划（灰度发布 or 记录不引入原因，定期复评）。

> 铁律：**公开 benchmark 高 ≠ 你的场景好**，必须用 ≥100 条真实业务数据 A/B；每次评估用同一脚本与数据，结果可复现；量化一切结论（"TTFT 25ms→18ms，P99 80ms→55ms"）。

---

## 附：FDE 现场速查决策

- 吞吐不够 → Continuous Batching + 量化 +（大流量）PD 分离。
- 延迟不够 → 投机解码（EAGLE-3，2–6x）/ 降 batch / FP8。
- 显存爆 → INT4 权重 + INT8 KV Cache（70B 可 1×A100：35+32=67GB）。
- 格式必须合规 → SGLang FSM。
- H100 已就位 → FP8 + EAGLE-3（改动最小、收益最大）。
- 不要盲目追新：PagedAttention + FP8 + Continuous Batching 已解决 80% 推理问题。
