> 由 SKILL.md 路由加载：模型架构、显存与吞吐公式、GPU 选型、瓶颈判定。
# 模型架构 + GPU 硬件基础 — FDE 部署蒸馏笔记

> 目标：给客户报机器配置、判断瓶颈、选型、排障时直接能用的数字与公式。删掉推导与科普。

---

## 1. 显存估算（报机器配置用）

### 1.1 模型权重大小
```
权重大小(GB) = 参数量 × bytes_per_param ÷ 10^9
bytes_per_param: FP16/BF16=2, INT8=1, INT4=0.5, FP8=1
```
- 70B FP16 = 140 GB；INT8 = 70 GB；INT4 = 35 GB
- 8B FP16 = 16 GB；INT8 = 8 GB

### 1.2 KV Cache 显存（最大动态显存消耗，占总显存 60–80%）
```
KV_Cache(GB) = 2 × num_layers × batch_size × seq_len × num_kv_heads × head_dim × bytes ÷ 10^9
说明：
  2 = K 和 V 各一份
  bytes: FP16=2, FP8/INT8=1, INT4=0.5
  GQA/MQA: num_kv_heads < num_q_heads（MHA 时等于 num_q_heads）
```

> **⚠️ 这一步最容易算错，实测中已出现过：不要用 hidden size（num_heads × head_dim）当 KV 维度。**
> - 错：Llama-70B 按 `2×80×8192×2` 算得 2.5MB/token → 高估 8 倍 → 显存结论和加卡建议全部失真。
> - 对：GQA-8 是 `num_kv_heads=8`、`head_dim=128` → `2×80×8×128×2` = 320KB/token。
> - 先查清该模型的 `num_kv_heads` 与 `head_dim` 再动手，不要拿 hidden size 顶替。
示例（Llama-3-70B，GQA-8，num_layers=80，num_kv_heads=8，head_dim=128，FP16）：
- batch=32, seq=8192 → ≈ 80 GB
- 单 token KV = 2×80×8×128×2 ≈ 320 KB
- GPT-3 175B 用 MHA（num_kv_heads=96）→ 单 token 1.5 MB，batch=32/seq=8192 ≈ 708 GB，基本无法大批量

**MHA→GQA 收益**：KV Cache 减 G 倍（G 为分组数）。G=8 时质量损失 <1%，KV Cache 减 8 倍，batch 上限提升 4–8 倍。

### 1.3 总显存分配（单请求期间）
| 组成 | 性质 | 量级（70B FP16, batch=32, seq=2048） |
|------|------|------|
| 权重 | 静态（启动固定） | 140 GB |
| KV Cache | 动态（随请求增长） | 20–40 GB（seq 越长越大） |
| 激活值 | 动态 | 2–5 GB |
| 预留/碎片 | — | 5–10% |

**结论**：70B FP16 最低 2×A100-80G（仅权重），推荐 4×A100-80G（含 KV Cache）。留 10–20% 给激活与碎片，不要全部分给 KV Cache。

### 1.4 估算吞吐（decode，瓶颈是带宽）
```
单卡 decode 吞吐(token/s) ≈ 显存带宽(GB/s) ÷ 权重大小(GB)
```
- 70B FP16 在 H100(3.35 TB/s)：3350÷140 ≈ 24 token/s（单卡理论上限）
- 4 卡 TP：每卡 35 GB → 3350÷35 ≈ 96，扣通信后实测 70–80 token/s
- INT8 量化：权重大小减半 → 速度近翻倍

---

## 2. 性能瓶颈判定（先判定再动手）

### 2.1 计算强度与 Roofline
```
计算强度 AI = FLOPs ÷ 内存访问字节数(bytes)
H100 拐点(平衡点) ≈ 989 TFLOPS ÷ 3.35 TB/s ≈ 295 FLOP/byte
  AI < 拐点  → Memory-Bound
  AI > 拐点  → Compute-Bound
```
快速判定（nvidia-smi）：GPU-Util >70% 多为 compute-bound；<30% 多为 memory-bound；两者都低可能是 IO-bound 或同步开销。

### 2.2 推理两阶段（最核心）
| 维度 | Prefill（处理 prompt） | Decode（逐 token 生成） |
|------|------|------|
| 复杂度 | Attention O(n²·d)，FFN O(n·d) | 每步 O(d²)，但要加载全部权重 |
| 计算强度 | 高（n 大时） | ~1 FLOP/byte（极低） |
| 瓶颈 | **Compute-Bound**（GPU 利用率 80–95%） | **Memory-Bound**（GPU 利用率 10–30%） |
| 时间占比 | 5–15% | 85–95% |
| 优化方向 | FlashAttention、Kernel Fusion | 量化、减权重、Batch 合并 |

**Decode 定量**（70B FP16）：每 token 计算 140 GFLOPs，但需加载 140 GB 权重；A100 上计算仅需 0.45ms，权重搬运需 70ms → 利用率 ~0.6%，纯 memory-bound。

**优化方向铁律**：memory-bound 时优化 kernel 无意义，必须做量化/减权重/增大 batch。

### 2.3 经验吞吐基准
- A100 decode 70B：8–15 token/s；H100：15–30 token/s（带宽 ×1.67）
- Decode 速度 = 总带宽 ÷ 权重大小，是物理上限，算法优化绕不过。

---

## 3. GPU 选型对照（按市面在售型号）

| 型号 | 架构 | 显存 | 带宽 | FP16 Tensor | FP8 | NVLink | 定位 |
|------|------|------|------|------|-----|--------|------|
| H100 SXM | Hopper | 80GB HBM3 | 3.35 TB/s | 989 TFLOPS | 1978 | 900 GB/s | 主力生产 |
| A100 SXM | Ampere | 80GB HBM2e | 2.0 TB/s | 312 TFLOPS | 不支持 | 600 GB/s | 主力通用 |
| B200 SXM | Blackwell | 192GB HBM3e | 8.0 TB/s | ~2.5 PFLOPS | ~5 PFLOPS | 1.8 TB/s | 高性能生产 |
| L40S | Ada | 48GB GDDR6X | 864 GB/s | 366 TFLOPS | 731 | 不支持 | 性价比推理 |
| RTX 4090 | Ada | 24GB GDDR6X | 1.0 TB/s | 330 TFLOPS | 660 | 不支持 | 开发/实验 |
| A800/H800 | — | 80GB | 接近 A100/H100 | 同 A100/H100 级 | — | 受限(400GB/s) | 中国市场合规版 [需确认具体互联带宽] |

> 注：A800/H800 为合规出口版，NVLink 带宽较原版受限，具体削减比例需按官方规格核对，标注 [需确认]。

**选型规则**
1. 显存容量决定能放多大模型（70B FP16≈140GB → 至少 2 卡 80G）。
2. 带宽决定 decode 速度（HBM ≫ GDDR：H100 比 4090 快 2–3× 即便同显存）。
3. 多卡 TP 必须 NVLink；4090/L40S 无 NVLink，只能单卡或走 PCIe（延迟差 10×）。
4. 精度支持：A100 不支持 FP8，FP8 量化只能在 H100/B200 上用 Tensor Core 加速。
5. 开发用 4090，生产用 A100/H100；SXM 版优于 PCIe 版（NVLink 仅 SXM）。

**场景速查**
- 70B+ 生产：H100/B200 ×8（NVLink 全互联）
- 7B–13B 生产：A100-80G ×2–4（性价比最优）
- 成本敏感 7B：L40S ×1–2
- 多租户小模型：A100 MIG（切最多 7 个隔离实例，不适合大模型 TP）
- 边缘 <1B：A10/L4

---

## 4. 多卡互联（张量并行可行性边界）

| 互连 | 双向带宽 | 延迟 | 适用 |
|------|---------|------|------|
| NVLink 4.0（同机） | 900 GB/s | 2–3 µs | TP（张量并行） |
| PCIe 5.0 x16 | 128 GB/s | 5–10 µs | PP/数据加载 |
| InfiniBand NDR（多机） | ~50 GB/s | 0.6 µs | DP（数据并行） |
| RoCE v2（多机） | 25–50 GB/s | 1–2 µs | DP/PP（非 TP） |
| TCP/IP 以太网 | — | >100 µs | 不可用 |

**铁律**
- TP 绝不跨机：跨机 TP 每层 All-Reduce，80 层总延迟 NVLink≈240µs vs IB≈1.2ms（差 5×）。
- 主流方案：TP 在单机内（NVLink），DP 跨机（IB/RoCE）。
- 4090 无 NVLink，只能单卡开发。
- RoCE 需无损网络（PFC + ECN + DCQCN），否则 RDMA 重传暴增延迟。
- H100 SXM 经 4 个 NVSwitch 实现 8 卡全互联，任意两卡 900 GB/s。

---

## 5. Transformer 组件对推理的实际影响

- **Attention 复杂度**：Prefill O(n²)，Decode O(n)；KV Cache 避免 O(n³) 重算（降到 O(n²)）。
- **KV Cache 大小**：见 §1.2。MHA 爆显存，GQA/MQA 必需。
- **FFN**：占每层参数 2/3–3/4（d_ff≈2–4×d_model，SwiGLU 有 3 个矩阵）。Prefill 时 FFN 占 ~73% FLOPs，因此**权重量化对首 token 延迟优化最明显**。
- **Normalization**：现代用 Pre-Norm + RMSNorm（无 β，计算少 ~7%），推理显存友好；Post-Norm 中间激活更大、batch 上限更低。
- **位置编码**：RoPE 主流（相对位置、可 NTK/YaRN 外推到训练长度 8–16×）。**超过训练长度 8× 质量明显下降**，部署需启用 YaRN/NTK-aware scaling。

**GQA 决策**：生产优先 GQA > MQA > MHA；seq_len>16K 必须用 GQA/MQA，MHA 的 KV Cache 会 OOM（batch=1 都可能 OOM）。

---

## 6. 特殊架构部署差异

### 6.1 MoE（Mixtral 8×7B / DeepSeek-V3 / Qwen-MoE）
- **显存由总参数决定，非激活参数**：Mixtral 8×7B 总参 46.7B→93GB，至少 2×A100-80G；DeepSeek-V3 671B 需 8+×H100。
- **推理不一定比 Dense 快**：每步加载全部 Expert 权重（memory-bound 更重）+ 跨卡 All-to-All 通信 + Router 开销。小 batch 下可能更慢。
- 部署：Expert 全放单卡最优；多卡需 NVLink（非跨节点）；激活频率偏差 >2× 说明负载不均。vLLM 对 MoE 支持不如 Dense 成熟。

### 6.2 MLA（DeepSeek-V2/V3/R1）
- 把 K/V 低秩压缩，只缓存 c_kv（512 维），KV Cache 比 GQA-8 再减 ~62%。
- DeepSeek-V3 单 token KV≈122KB（GQA-8 同档 320KB）；batch=32/seq=4096 仅 12.5GB。
- 量化特殊：c_kv 分布异于标准 K/V，需针对校准；**FP8 推荐（H100 1.5–2×加速），INT4 不推荐**。
- 仅 DeepSeek 系列适用；TensorRT-LLM 支持不全，推荐 vLLM。

### 6.3 多模态（LLaVA / Qwen-VL 等）
- 视觉编码器开销 <5ms（可忽略），瓶颈仍在 LLM。
- 视觉 token：1 张图≈256 token，4 张≈1024，10 张≈2560 → 直接加进 KV Cache。
- 显存增 25–30%，batch 减 20–30%，首 token 增 ~50ms。
- KV Cache = 文本 KV + 视觉 KV（视觉部分用 num_patches 替代 seq_len）。
- 最易优化：**图片特征缓存**（相同图不重复编码，省 90%）、动态分辨率、批量编码。

### 6.4 Thinking / 推理模型（R1/o1/o3）
- 输出长度暴涨 20–85×（思考 2000–8000 token + 答案）。KV Cache、GPU 时间、成本同比例涨（参考量级：H100 上 70B 月成本 ~$3000→~$40000，约 13×，[需确认]）。
- 单 GPU QPS 从 10–50 降到 1–5。
- KV Cache 策略：全缓存（单用户）/滑动窗口/丢弃思考（单次问答）/压缩思考（高价值）。
- 优化：分层路由（简单问题走非 thinking）、答案缓存、限制 max_thinking_tokens、Prefill-Decode 分离（decode 节点按需扩容）。
- API 需区分 thinking/answer 流式输出（stream_options.separate_thinking）。

---

## 7. 训练与微调（决定部署方式）

### 7.1 何时该微调
- 领域适配（医疗/法律 10K–50K 条）、格式调整（1K–5K 条）、复杂推理（50K+ 条）。
- 数据质量 > 数量：1K 高质量 ≈ 50K 低质量。

### 7.2 LoRA vs 全参 vs QLoRA
| 方法 | 70B 显存需求 | 效果 | 适用 |
|------|------|------|------|
| 全参微调 | ~560GB（权重140+梯度140+优化器280；需 8×A100） | 最好 | 数据>100K、单任务 |
| LoRA（r=16） | ~40–175GB（基座可 INT4→35GB） | 接近全参 | 显存≥80GB |
| QLoRA（4-bit + LoRA） | 48GB(r=16) / 52GB(r=64) | 与 LoRA 差 <1% | 单卡 A100 即可 |

- LoRA 参数量占比 ~0.06%（70B 约 42M）；rank r=8–64，alpha=2×r，target=q_proj,v_proj。
- 部署：单任务合并权重（无开销）；多任务用 vLLM/SGLang LoRA Hub（额外开销 <2%，热切换 <1ms）。
- 选择规则：显存≥80GB 且数据>100K→全参；显存紧张→QLoRA；显存 16–40GB→QLoRA。

### 7.3 Scaling Law（报训练算力用）
```
训练总 FLOPs ≈ 6 × N × D   （N=参数量, D=训练token数；前向2×+反向4×）
```
- 70B × 15T token ≈ 6.3×10²⁴ FLOPs；H100 有效算力 ~396 TFLOPS（MFU 40%）→ 约 3000 卡训练 60 天。
- Chinchilla：N_optimal ∝ D^0.73，数据不足时堆参数收益递减。推理成本 ~$0.0003/1Ktok(7B) → ~$0.003(70B) → ~$0.017(405B)。

---

## 8. 踩坑清单 + 验收标准

**排障表**
| 症状 | 原因 | 处理 |
|------|------|------|
| 长 prompt 后 OOM | KV Cache 超显存 | 查 batch×seq×num_kv_heads；降 max_model_len |
| decode 极慢 | memory-bound | 开 FlashAttention、量化、TP |
| 首 token 高延迟 | prefill 量大 | chunked prefill / 前缀缓存 |
| 吞吐不稳 | batch 动态变化 | 开 continuous batching |
| OOM 但显存够 | 连续分配碎片 | PagedAttention（利用率 60%→95%） |
| MoE 比 Dense 慢 | 跨卡 All-to-All | 优先 NVLink、单卡放 Expert |
| 长上下文质量降 | RoPE 外推不足 | 启 YaRN/NTK-aware |

**部署验收标准**
- KV Cache 使用率 60–90%，>95% 需扩容/限并发；碎片率（PagedAttention）<5%，>10% 查 block_size。
- gpu_memory_utilization 设 0.85–0.95，留激活与 CUDA context。
- FlashAttention 必开（Ampere+），GQA 默认 8 groups，continuous batching 必开（提吞吐 2–3×）。
- chunked prefill：prompt>8K 启用；prefix caching：多轮/Agent 场景必开（省 50–90% KV）。
- 监控 GPU-Util：decode <30% 正常（memory-bound），不要误判为故障。

---

## 9. 关键公式速查
```
权重大小(GB)        = 参数量 × bytes_param ÷ 10^9
KV Cache(GB)        = 2 × layers × batch × seq × kv_heads × head_dim × bytes ÷ 10^9
单卡 decode 吞吐     ≈ 带宽(GB/s) ÷ 权重大小(GB)
计算强度 AI         = FLOPs ÷ bytes
训练 FLOPs          ≈ 6 × N × D
decode 速度上限      = 总带宽 ÷ 权重大小  （不可被算法绕过）
```
