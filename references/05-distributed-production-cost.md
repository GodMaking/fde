> 由 SKILL.md 路由加载：并行策略、生产架构、可观测、成本与容量。
# FDE 技能蒸馏：分布式推理 + 生产部署 + 成本与容量

> 适用场景：FDE 在企业客户现场做 LLM 推理方案——需求拆解、技术选型、方案设计、POC、部署落地、成本测算、复盘推广。本文只保留能直接用的判断阈值、公式、选型表、命令/配置、决策规则、踩坑清单、验收标准。模型规模与吞吐数字为公开基准值，落地前必须按客户实际硬件压测复核，文中以 [需确认] 标注的价格/型号随市场变动。

**本文件覆盖的现场能力**：① 给定模型规模和硬件，选对并行策略并算出卡数；② 照清单搭出可上线的生产架构（网关/多租户/弹性/容灾/PD 分离）；③ 配齐可观测指标与告警，5 分钟内定位延迟故障；④ 给客户算出单 token 成本、容量与自建/云决策；⑤ 避开方案评审中最常见的 10 个坑。

---

## 一、并行策略选型（TP / PP / EP / SP）

### 1.1 四种策略各解决什么

| 策略 | 切分维度 | 解决的核心问题 | 通信原语 | 通信频率 | 必须的网络 | 典型场景 |
|------|----------|----------------|----------|----------|------------|----------|
| DP 数据并行 | 模型完整复制，不同请求不同卡 | 提升吞吐，不解决放不下 | 无 | 0 | 无 | 模型 < 单卡显存 |
| TP 张量并行 | 层内权重矩阵 | 单卡放不下、延迟最低 | AllReduce | 每层 2 次 | **NVLink 域** | 7B–70B，单节点内 |
| PP 流水线并行 | 按层 | 超大模型跨节点部署 | Send/Recv | 每 micro-batch 1 次 | PCIe/跨机均可 | 70B+，跨节点 |
| EP 专家并行 | Expert 分布 | MoE 模型显存/计算分散 | AllToAll | 每 MoE 层 2 次 | 高带宽跨卡 | MoE（DeepSeek-V3 等） |
| SP 序列并行（即 Context Parallel） | 序列维度 | 超长上下文显存 | AllGather/ReduceScatter | 跨卡 attention | 高带宽 | 128K+ tokens |

**核心矛盾与判断顺序**：分布式推理的核心矛盾是——计算可以线性切分，但通信开销不是线性的；选错策略，8 张卡的性能可能不如 4 张。判断顺序永远固定为三步：先看「单卡放不放得下」→ 再看「放不放得进单 NVLink 域（单节点 8 卡）」→ 最后才看「跨不跨节点」。单卡放得下就别并行，用 DP 复制提吞吐最稳、零通信；放得进 NVLink 域优先 TP；只有超出单节点才上 PP/EP 跨机。

### 1.2 决策规则与阈值（按优先级）

**先判断「单卡能否放下」：**
- 单卡放得下（如 7B/13B/70B + INT4）→ 用 DP 提吞吐，无需切分。
- 放不下，但能放进**单 NVLink 域**（如单节点 8×A100/H100）→ TP 优先。

**单卡放不下时的硬性阈值：**
- TP size 取 2 的幂（2/4/8）；**TP 一般不超过单节点 GPU 数（默认 ≤8）**，因为跨机 AllReduce 延迟是 NVLink 的 50–100 倍，基本不可用。
- 跨机才能放下 → 用 PP 跨节点 + TP 节点内（TP+PP+DP）。
- MoE 架构 → EP + 可能的 TP；按**激活参数**规划，而非总参数（DeepSeek-V3 总参 671B，激活仅 ~37B）。

**单卡放不下先加 TP 还是先量化**（明确的二选一规则）：
- 前提：单卡显存放不下 FP16 权重。
- 规则：先**量化**（INT8 省 ~50% 显存、INT4 省 ~75%）把模型压进单节点 NVLink 域；仍放不下（如 70B FP16 需 140GB）再上 TP。
- 理由：TP 跨 PCIe/跨机通信开销陡增，能靠量化避免就不上 TP；量化几乎不增通信、优先于切分。

**「跨机该怎么切」：**
- 跨机 = PP（相邻卡 Send/Recv 激活值，通信量 ~1MB/micro-batch，PCIe 即可，16μs 级）或 EP（MoE 专用，AllToAll 可批量）。
- **禁止跨机做 TP**：TP 每层 2 次 AllReduce 直接叠加到每 token 延迟。

### 1.3 通信开销与延迟/吞吐影响

| 策略 | 单卡每次通信量（Llama-3-70B 参考） | 扩展效率（NVLink→PCIe→跨机） | 对延迟影响 |
|------|--------------------------------------|-------------------------------|------------|
| TP | 每层 ~32KB（AllReduce），~56KB/层@TP=8 | H100 92% / A100 85% / PCIe 45% / IB 15% / 25G RoCE 2% | 极高敏感，逐层叠加 |
| PP | ~1MB/micro-batch（Send/Recv） | 跨机可用，有 bubble | 低敏感，被 bubble 掩盖 |
| EP | ~16MB/MoE 层（AllToAll），占总延迟 30–50% | 跨机可用 | 中等，可批量 |

- AllReduce 每卡发送量 = `2×(P-1)/P×M`：TP=4 发 1.5×、TP=8 发 1.75× 数据量，scaling 效率随 P 递减。
- Pipeline Bubble 比例 = `(P-1)/(M+P-1)`（P=stage 数，M=micro-batch 数）。P=4,M=4 时 43%；增大 M 可降；推理无 backward，靠 continuous batching 填 bubble。
- **实战结论**：TP 的 scaling 效率随卡数递减且对网络极度敏感，能用量化避免跨机就避免；PP 的 bubble 是可接受的代价，换来跨节点部署能力；EP 的 AllToAll 是 MoE 主瓶颈，但跨机比 TP 友好（频率低、可批量）。选策略的本质是在「延迟、吞吐、跨节点能力」三者间取舍，不是越大并行越好。

### 1.4 常见组合方式
- TP + DP：NVLink 内 TP（2–8 卡一组），域间 DP 多副本提吞吐（最常见）。
- TP + PP + DP：超大模型，PP 跨节点、TP 节点内。总 GPU 数 = TP × PP × DP。
- MoE + EP + TP：EP 分布 Expert，Expert 内再 TP（Expert 本身大时）。总 GPU = TP × EP × PP。

### 1.5 MoE 特殊性
- 按激活参数规划：DeepSeek-V3 激活 37B → 按 37B 量级配 TP/PP。
- AllToAll 通信是主瓶颈（占延迟 30–50%），缓解：通信聚合、热门 Expert 复制、Token Batching、计算-通信 overlap。
- 负载均衡：推理无 aux loss，靠 capacity_factor（1.0–1.25，溢出走 shared expert）+ 动态重路由。
- 部署命令参考：`vllm serve deepseek-ai/DeepSeek-V3 --tensor-parallel-size 4 --enable-expert-parallel`。

**显存估算（决定单卡能否放下，先算再选策略）：**
- 单卡显存 ≈ (参数量 ÷ TP_size) × 每参数字节数 + KV_Cache×batch + 激活值 + 通信缓冲
- KV_Cache = 2 × num_layers × num_heads × head_dim × seq_len × batch × 2字节（系数 2 为 K/V 两份）
- 判定阈值：7B FP16 ≈14GB（A100-80G 单卡可放）；70B FP16 ≈140GB（需 TP=2 或 INT4 ~35GB 单卡可放）；671B FP16 ≈1.3TB（必须多节点 + 量化）。
- 经验：单卡放不下先试 INT4/FP8 压进单节点 NVLink 域；压不进才上 TP/PP。

**启动命令速查（vLLM）：**
| 场景 | 命令 |
|------|------|
| 7B 单卡 | `vllm serve <model>` |
| 70B 单节点 | `vllm serve <model> --tensor-parallel-size 8` |
| 跨节点 TP+PP | `--tensor-parallel-size 4 --pipeline-parallel-size 2` |
| MoE EP | `vllm serve DeepSeek-V3 --tensor-parallel-size 4 --enable-expert-parallel` |

**PP vs TP 选型条件**：选 PP 当模型超单 NVLink 域、跨节点部署、异构集群、需与 TP 组合；选 TP 当可入单节点、延迟敏感、负载均衡要求高。PP 扩节点、TP 锁节点内。

**验收标准**：单节点内 TP 扩展效率 ≥85%（H100）/≥80%（A100）；跨机不得用 TP；MoE 全链路无单 Expert 持续 >90% 利用率（均衡）。

---

## 二、生产部署架构

### 2.1 能直接照着搭的架构清单（组件选型）

| 层 | 组件 | 选型 | 关键要求 |
|----|------|------|----------|
| 边缘 | CDN/WAF | 标准 | 安全防护 |
| 网关 | API Gateway | Kong / APISIX | SSE 流式稳定、按模型路由、按 API Key 限流（按 token 而非请求数） |
| 负载 | LB | Nginx/HAProxy/ALB | 算法用 `least_conn`（LLM 请求时长差异大），健康检查 `/health` |
| 推理 | Inference Service | vLLM / TRT-LLM / SGLang / TGI | Continuous Batching、PagedAttention/RadixAttention |
| 存储 | Model Registry | S3 / MinIO + MLflow | 大文件分发、蓝绿双版本并存 |
| 可观测 | 监控 | Prometheus + Grafana + Loki + Jaeger + DCGM Exporter | 见第三节 |
| 编排 | K8s | NVIDIA Device Plugin + GPU Operator | 节点亲和/反亲和、Taint/Toleration、PriorityClass |

**架构选型总原则**：边缘层做轻、网关层做限流与路由、推理层只管算、存储与可观测独立部署。所有 LLM 特有能力（KV Cache 感知路由、Token 级限流、SSE 流式、按 model 字段路由、前缀缓存）必须在网关或推理引擎层解决，不要把 LLM 语义硬塞进普通 LB。推理 Pod 之间不互相依赖状态，模型权重走独立 Model Registry，监控链路独立可观测——这样任一 Pod 挂了不影响其他副本。

- K8s 关键配置：`maxUnavailable:0 + maxSurge:1`（零停机滚动更新）；`readinessProbe.initialDelaySeconds ≥ 120`（70B 加载 60–90s）；`/dev/shm` ≥16GB（NCCL，默认 64M 不够）；Pod 反亲和分散到不同主机。
- GPU 节点：装 NVIDIA Container Toolkit + Device Plugin；Taint `nvidia.com/gpu=:NoSchedule` 防非 GPU Pod 抢占。
- 模型加载时间线（70B AWQ INT4 @ 4×A100）：下载 30s + 显存加载 20s + KV 分配 5s + warmup 3s ≈ **60s**（冷启动全链路约 80s）。

### 2.2 推理网关（何时需要、怎么选）
- 传统 LB（Nginx round_robin/least_conn）问题：请求时长差异 50ms–2000ms、忽略 KV Cache 会话状态、不懂 model 字段/限流按请求数。
- 必备能力：KV-Cache 感知路由（session→instance 映射存 Redis，多轮 TTFT 降 90%+）、前缀感知路由（相同 system prompt 命中 RadixAttention，prefix prefill 从 100ms→<5ms）、模型路由、负载感知（按 GPU 利用率+KV 使用率而非连接数）、Token 级限流（Redis+Lua 滑动窗口）。
- 选型决策：单模型 QPS<50 → Nginx/vLLM Proxy；SGLang 引擎 → 内置 Router（原生前缀感知）；多模型大规模/云原生 → Envoy + LLM-D；需细粒度 KV 路由 → 自研 + Redis。
- SSE 配置：Envoy `timeout 300s / idle_timeout 60s`，否则长连接被断。

### 2.3 多租户隔离
| 隔离方式 | 隔离度 | 成本 | 适用 |
|----------|--------|------|------|
| 独占 GPU | 完全 | 高 | 核心生产 |
| MIG 分片（A100/H100，最多 7 片/卡） | 中（硬件级显存/算力） | 中 | 多租户中等负载；**不支持 NVLink，不能跑 TP** |
| Time-Slicing | 低（软件级） | 低 | 开发/测试 |
| Namespace + ResourceQuota + PriorityClass | 最低 | 最低 | 内部团队 |

- 防护清单：ResourceQuota 限 GPU 总数；API Gateway 按租户限 QPS；单请求超时 + prompt 长度上限（防占满 KV Cache）；PriorityClass 核心不可抢占、批量可抢占；NetworkPolicy 租户间网络隔离；计费埋点 `input_tokens/output_tokens/gpu_seconds`。
- MIG 限制：每个 MIG 实例独立、无 NVLink，**70B+ 大模型必须独占多卡，不能用 MIG**。
- **MIG 用法经验**：适合把一张 A100 切成 7 个 `1g.5gb` 跑 7 个 7B 租户，最大化单卡利用率；也可 `3g.20gb + 4g.40gb` 混合让生产独占、测试共享。但 MIG 实例间无 NVLink 互联，无法做 Tensor Parallel——凡需要多卡切分的 70B+ 模型一律独占整卡或多卡，绝不能用 MIG 跑大模型。

### 2.4 弹性伸缩
- 标准 HPA（CPU）不适用：LLM 是 GPU-bound，CPU 30% 时 GPU 已 100%；冷启动 30s–3min。
- 用 KEDA 基于自定义指标：`vllm_num_requests_waiting > 10`、GPU 利用率 `> 85%`、`vllm_gpu_cache_usage_perc > 80%`、P99 `> 2s`。
- 关键参数：`cooldownPeriod 300s`（缩容保守）；`minReplicaCount` 保 Warm Pool（始终 2 个热 Pod，消除冷启动）；`scaleDown stabilizationWindow 600s`、每次只缩 1 个。
- 冷启动缓解：Warm Pool、预测性扩容（Cron Scaler 提前 5–10min）、镜像预打包权重、init container 预拉取、模型预热 dummy inference。
- **冷启动是弹性最大的敌人**：新 Pod 从调度到接流量有 60–80s 空窗，单靠 HPA 必然超时丢请求。应对分三层——事前 Warm Pool + 预测扩容保底；事中 API Gateway 先限流/排队、KEDA 触发扩容；极端时降级切小模型或异步队列。突发 10 倍时扩容来不及，先立限流再切小模型（7B 单卡可起更多副本）。
- 优雅退出：`terminationGracePeriodSeconds ≥ 最长推理时间×2`，preStop 通知 LB 停止分发 + 等请求完成再 SIGKILL。

### 2.5 容灾与高可用
- 故障分级：硬件（Xid Error/ECC/过热）、软件（OOM Killed/显存泄漏/死锁）、数据（权重损坏/KV 污染）、外部（S3/DNS 不可用）。
- Xid Error 自动隔离：DCGM 每 15s 采集 → Prometheus 告警 `increase(XID_ERRORS[5m])>0` → AlertManager Webhook `kubectl cordon+drain` → 健康节点重建 Pod（~60s 加载）→ 恢复（总 0–180s）。Xid 63/79 立即隔离换卡。
- 多 AZ：流量 80/20 或 50/50，故障切换 <30s；S3 跨区域复制 + 启动校验 SHA256；Pod `topologySpreadConstraints maxSkew:1` 跨 zone。
- 四级降级：L1 减小 batch（256→64）；L2 切小模型（70B→7B，独立资源池防雪崩）；L3 限流+排队（核心白名单）；L4 规则引擎兜底（不依赖 GPU）。
- 模型回滚：`kubectl rollout undo`；保留上一版本镜像 5 分钟内可回滚。

### 2.6 PD 分离部署
- 原理：Prefill 是 compute-bound（GPU 利用率 80–95%），Decode 是 memory-bound（10–30%），混部互相拖累、扩缩不灵活。
- 收益（Llama-3-70B @ A100×8 参考）：QPS 5→12（2.4x），TTFT/TPOT 各降 ~40%。
- 方案选型：QPS<10 不值得；单实例/中小流量 → ChunkFuse（vLLM 配置，1.3–1.5x，零额外网络）；大流量独立集群 → DistServe（2x+，需 RDMA）；多租户弹性 → MoonCake（KV Cache 池化，1.5–2x）。Thinking 模型 Prefill 极短/Decode 极长，PD 分离收益最大。
- 决策阈值：**QPS > 10 且有独立 GPU 集群 + 资源需求差异大**才上完整 PD 分离。

### 2.7 部署验收清单（上线前逐项勾）
- 滚动更新 `maxUnavailable:0 + maxSurge:1`，`readinessProbe.initialDelaySeconds ≥ 模型加载时间`（70B ≥120s）。
- `/dev/shm ≥16GB`（默认 64M 会让 NCCL 跨进程通信失败）；Pod 反亲和分散到不同主机，避免单点故障。
- 模型权重镜像预打包或 init container 预拉取，下载 ≤30s；蓝绿发布需双倍 GPU 资源。
- 金丝雀 5%→25%→50%→100%，每步观察 10–30min，重点盯 TTFT 与输出质量（LLM-as-a-Judge 自动校验）。
- 上线前 Grafana Dashboard + AlertManager 规则已生效；回滚预案：保留上一版本镜像，5 分钟内可 `kubectl rollout undo`。
- 多租户：ResourceQuota 限 GPU 总数 + PriorityClass 核心不可抢占 + API Gateway 按租户限流 + prompt 长度上限。

---

## 三、可观测性

### 3.1 必须采集的指标清单（三层）
- **业务层**：TTFT（P50<200ms / P95<400ms / P99<800ms）、TPOT（P99<100ms）、端到端延迟、请求队列长度 `vllm_num_requests_waiting`、KV Cache 使用率 `vllm_gpu_cache_usage_perc`、活跃请求数、错误率 `vllm_request_success_total`。
- **引擎层**：Request/s（70B ~2–5/GPU·s）、Token/s（70B ~50–100/GPU·s）、Batch Size。
- **GPU 层**（DCGM）：`GPU_UTIL`（健康 70–95%）、`FB_USED` 显存（>95% OOM 前兆）、温度（>85℃ 告警）、功耗（>90% TDP）、`NVLink RX/TX`、`XID_ERRORS`（>0 立即告警）。
- 日志：结构化 JSON，含 `request_id/trace_id/model/prompt_tokens/completion_tokens/ttft_ms/status`；FluentBit→Loki，`max_line_size 64KB`（长 prompt）。
- Tracing：OpenTelemetry + Jaeger，Span 覆盖 Gateway→Router→排队(TTFT)→Prefill→Decode，定位慢在哪一阶段。
- **三层指标的使用分工**：Metrics 回答「系统是否健康」用于告警与趋势；Traces 回答「慢在哪里」用于定位瓶颈；Logs 回答「具体发生了什么」用于根因。三者缺一不可——只配 Metrics 会知道告警但查不出原因，只配 Logs 会在海量日志里溺水。

### 3.2 告警阈值（直接抄）
| 告警 | 表达式 | 级别 |
|------|--------|------|
| TTFT P99 | `histogram_quantile(0.99, rate(ttft_bucket[5m])) > 0.8` | warning；>2.0 critical |
| GPU 显存 | `FB_USED/(FB_USED+FB_FREE) > 95%` | critical |
| GPU 温度 | `> 85℃` 持续 5m | warning |
| Xid Error | `increase(XID_ERRORS[5m]) > 0` | critical |
| 队列堆积 | `sum(vllm_num_requests_waiting) > 50` 持续 3m | warning |
| 错误率 | `error/(total) > 5%` 持续 2m | critical |
| KV Cache | `avg(vllm_gpu_cache_usage_perc) > 0.9` | warning |

### 3.3 典型故障排查路径（P99 TTFT 飙升）
1. Grafana 看 GPU 利用率：≈100% → 扩容；<30% → 异常（查 Xid/模型权重）。
2. 队列长度：长 → 超负载，扩容/限流；正常 → 单请求变慢。
3. KV Cache 使用率 ≈100% → 大 prompt 占满，调 `max_num_seqs`。
4. GPU 低利用率+Xid → `dcgmi diag` 隔离节点迁移 Pod。
5. 单请求慢 → Jaeger 看 Prefill 慢（prompt 太长）还是 Decode 慢（功耗/降频）。
根因 TOP5：流量突增、GPU 硬件故障、KV Cache 饱和、模型权重损坏（回滚）、同卡有其他进程。

### 3.4 上线前可观测性就绪清单
- Prometheus `scrape_interval 15s`，同时采集 vLLM `/metrics` + DCGM Exporter + node-exporter。
- Grafana 四大核心面板：P99 TTFT、GPU 利用率、GPU 显存使用率、请求队列长度（阈值黄 10 / 红 50）。
- AlertManager 至少配四条：TTFT P99>0.8s、显存>95%、Xid Error>0、队列>50。
- OpenTelemetry 链路打通，慢请求能区分在 Prefill 还是 Decode 阶段。
- 结构化日志含 `request_id/trace_id/model`，Loki 可按 Pod/请求检索；单行上限 64KB 适应长 prompt。
- GPU 利用率健康带：**70–95%**；<30% 是浪费（请求不足或 batch 太小，应减实例/增大 batch）；>98% 将 OOM 或排队，应扩容。

---

## 四、成本与容量

### 4.1 成本构成
推理总成本 = GPU 租赁 60–80% + 电力 5–10% + 网络 5–10% + 存储 3–5% + 运维人力 10–20%。**GPU 是优化第一对象。**

**给客户报价的底线**：① 永远用实测单卡吞吐反推，不轻信厂商理论峰值（H100 decode 单请求 1000 tok/s 只是上限，Continuous Batching 下实际 70–85%）；② 把 GPU 占 60–80% 讲清楚，优化焦点放在 GPU 而非电费；③ 月调用量超过 1 亿 token 必须给自建 TCO 与盈亏平衡点，不能只报 API 单价（大规模 API 比自建贵 15–20 倍）；④ 报价区分「日常成本」与「峰值弹性成本」，峰值用 Spot/云弹性，避免客户为全年峰值买单。

### 4.2 单 token 成本公式与构成
```
每 1K token 成本 = GPU 小时费 ÷ (单卡每小时处理 token 数 ÷ 1000)
```
- 例（H100 70B FP16，decode 150 tok/s）：`$12.29/h ÷ (150×3600/1000) = $0.023/1K`（无优化）；Continuous Batching 后 5x 吞吐 → `$0.0046/1K`。
- 量化对单 1K token GPU 成本影响（70B 参考）：FP16 `$0.08` → FP8/INT8 `$0.05`（省 ~67%）→ INT4 `$0.04`（省 ~73%）。
- 云 API 参考单价 [需确认，随厂商变动]：GPT-4o-mini `$0.00015` in / `$0.0006` out per 1K；Claude Sonnet4 `$0.003` / `$0.015`；Gemini Flash 更低。

### 4.3 自建 vs 云 API 决策规则
| 月处理 token | 方案 | 前提 |
|--------------|------|------|
| < 100 万 | Serverless API | 零运维、快验证 |
| 100 万 – 1 亿 | 云 GPU IaaS 或混合 | 有合规要求选云 GPU |
| > 1 亿 | 自建集群 | 有 GPU 运维团队 |

- 切换信号：月 API 费用 > `$5,000` 且持续增长 → 评估自建；月 > 2 亿 tokens 自建开始省钱。
- 盈亏平衡点 = 自建启动成本 ÷ (Serverless 单价 − 自建单位成本)。例：H100 服务器 `$50K`、API `$0.01/1K`、自建 `$0.002/1K` → 6.25B tokens 回本（月 >200M 约 6 个月回本）。
- 3 年 TCO 参考（70B，5 亿 token/月）：Serverless `$90M` ≫ 云 GPU 按需 `$20M` ≈ 自建 A100 `$4.4M` / H100 `$4.8M` / 混合 `$5.6M`。**大规模自建比 API 省 15–20 倍，但启动成本 $35K+ 且运维高。**
- 推荐混合 70/20/10：70% 自建基础负载 + 20% 云 GPU Spot 峰值 + 10% Serverless 降级兜底；路由按队列深度动态溢出。
- 混合路由的实现要点：在 API Gateway 监控「自建队列长度 + P99 + GPU 利用率」三个信号，超过阈值即溢出到云 GPU，回落时逐步切回避免震荡；自建队列 >N 且 P99>目标 → 溢出，云也满 → 降级 Serverless。保持推理接口抽象，随时可换后端。

### 4.4 容量规划公式（反推 GPU 数）
```
所需 GPU 数 = ceil(峰值 QPS × 平均输出 tokens ÷ 单卡 decode 吞吐) ÷ 利用率系数
```
- 单卡 decode 吞吐（H100 参考，Continuous Batching）：7B batch=16 ~5,500 tok/s；70B batch=16 ~550 tok/s；175B ~220 tok/s。
- 集群效率系数：单卡 100%、多卡 TP 90–95%、多节点 TP 80–90%。
- 例：峰值 600 QPS × 200 out tok ÷ 5,500（7B） = 22 卡；70B 则需 220+ 卡。**模型选择是容量第一决定因素。**
- 公式分子用「峰值 QPS × 平均输出 tokens」而非输入——decode 是逐 token 生成、占绝大多数耗时，输入只在 prefill 一次性消耗。容量瓶颈在 decode 吞吐，不在 prefill；压测必须按真实输出长度分布，不能只看 input。
- Buffer：正常 +30%、峰值 +50–100%、容灾 N+1。组合：60% 预留 + 25% 按需 + 15% Spot。
- SLA 驱动：严格 P99<500ms → GPU 超配 2.0x、batch 4–8、利用率 40–50%、N+2；宽松 P99<2s → 超配 1.3–1.5x、batch 16–32、利用率 70–85%、N+1，**省 40–50%**。

**容量规划算例（峰值 600 QPS，平均 200 out tok，H100，含 +30% 缓冲与 N+1）：**
- 7B（batch=16，单卡 5,500 tok/s）：600×200÷5,500 = 22 卡 → 约 30 卡。
- 70B（batch=16，单卡 550 tok/s）：600×200÷550 = 219 卡 → ÷0.85 利用率 + N+1 ≈ 259 卡。
- 严格 SLA 再 ×2 超配：7B ≈ 60 卡、70B ≈ 520 卡。
- 结论：模型降一档（70B→13B）比堆 GPU 便宜数倍；优先 Router 模式（70% 小模型 + 30% 大模型）把 70B 卡数压到 1/3。

### 4.5 成本优化手段与收益量级（累计可降 80–95%）
| 层级 | 手段 | 收益 | 质量影响 |
|------|------|------|----------|
| 模型层 | 选对尺寸（80% 请求用 7B–13B；Router 模式 70%小+30%大） | 省 30–53% | 路由兜底无损 |
| 引擎层 | Continuous Batching | 省 50–70%（吞吐 2–5x） | 无损 |
| 模型层 | 量化 INT8（省 50%）/INT4（省 70%，AWQ<0.3%） | 30–50% | INT4 损 1–3%，慎用于数学/代码 |
| 缓存层 | 语义缓存（命中 30–60% 客服场景，GPU 成本=0）+ 前缀缓存 | 5–40% | 无损 |
| 基础设施 | 预留+按需+Spot 混合 / 夜间降配 | 40–56% | Spot 中断 1–5% 需回退 |
| 引擎层 | Speculative Decoding | decode 2–3x | 基本无损 |

- 优先级：先选模型尺寸 → Continuous Batching → 量化 → 缓存 → 弹性+Spot。**先做 PTQ 量化再上线比上线后改简单 10 倍。**
- 累计收益（剩余成本占比）：基线 100% → 选对模型 50–70% → +Continuous Batching 15–35% → +INT8 量化 8–25% → +语义缓存 5–20% → +弹性/Spot 5–20%。**综合可降到基线的 5–20%（即省 80–95%）。**

---

## 五、给客户做方案时的常见坑

下面 10 条是客户方案评审中最常被忽略、且一旦出错就导致「上线即事故」的点，按现场出现频率排序。每条都对应一个可验证的验收动作，写进方案交付清单即可避险。

1. **压测不真实**：用理论峰值吞吐（如 H100 decode 1000 tok/s 单请求）算容量 → 实际 Continuous Batching 下只有 70–85%，且混部/跨节点再打 8–10 折。**必须 vLLM benchmark + k6/Locust 端到端压测后规划，按实测单卡吞吐反推。**
2. **没算模型加载/冷启动时间**：滚动更新、突发扩容都有 60–80s 加载窗口，容量规划不留 Warm Pool + 预测扩容 → 峰值期请求排队超时。**预留 minReplica 热 Pod，突发前 Cron 扩容。**
3. **多租户互相抢占**：只做 Namespace 逻辑隔离、无 ResourceQuota/MIG/PriorityClass → 某租户长 prompt 占满 KV Cache 拖垮全员。**硬隔离核心业务，限 prompt 长度 + 限流 + PriorityClass 抢占。**
4. **冷启动超时**：readinessProbe `initialDelaySeconds` 不够（70B 需 ≥120s）→ Pod 未加载完就接流量报错。**按模型大小设 ≥ 加载时间，/dev/shm ≥16GB。**
5. **TP 跨机**：为省钱把 TP 跨 PCIe/跨机 → 每层 AllReduce 延迟叠加，8 卡性能不如 4 卡。**TP 锁在单 NVLink 域，跨机只用 PP/EP。**
6. **KV Cache 饱和无预案**：大 prompt 把 cache 打满开始 reject → 队列雪崩。**监控 `vllm_gpu_cache_usage_perc>0.9` 告警，调 `max_num_seqs`，前缀缓存复用。**
7. **成本估算用 API 单价外推大规模**：月亿级 token 仍按 API `$0.01/1K` 报预算 → 实际贵 15–20 倍。**超过 1 亿 token/月 必算自建 TCO 与盈亏平衡点。**
8. **忽略 MIG 不能跑 TP**：给 70B 分 MIG 多实例 → 无 NVLink 无法张量并行、放不下。**70B+ 必须独占多卡 TP，MIG 只用于 7B/13B 多租户。**
9. **降级雪崩**：小模型兜底池被 70B 挤占 → 降级时小模型也挂。**小模型独立资源池，降级同步收紧限流。**
10. **SLA 与成本脱节**：严格 P99<500ms 配高 batch → 既超延迟又浪费。**严格 SLA 用低 batch(4–8)+2x 超配；宽松 SLA 用大 batch 省 40–50%。**

---

## 六、决策速查（阈值速记）

下面把全文阈值浓缩成一页速查表，客户现场或方案评审时直接对照打钩。所有数字都来自上文各节，落地前务必用客户真实硬件压测复核吞吐与加载时间。

| 决策点 | 阈值 / 规则 |
|--------|-------------|
| 单卡能否放下 | 7B≈14GB / 70B≈140GB(FP16)或 35GB(INT4) / 671B≈1.3TB；放不下先量化后进 NVLink 域 |
| TP size 上限 | ≤ 单节点 GPU 数（默认 8），必须 NVLink 域内，跨机禁用 |
| TP 扩展效率 | H100 NVLink 92% / A100 85% / PCIe 45% / 跨机 IB 15% / 25G RoCE 2% |
| 跨机切分 | 只用 PP（Send/Recv）或 EP（AllToAll），AllToAll 占 MoE 延迟 30–50% |
| MoE 规划 | 按激活参数（DeepSeek-V3 激活 37B），capacity_factor 1.0–1.25 |
| 模型加载/冷启动 | 70B INT4 @4×A100 ≈60s，全链路冷启动 ≈80s；minReplica 保 Warm Pool |
| 网关选型 | QPS<50 用 Nginx；SGLang 用内置 Router；大规模/云原生用 Envoy+LLM-D；KV 路由自研+Redis |
| 多租户隔离 | 核心独占 / 中载 MIG（无 NVLink，不能 TP）/ 开发 Time-Slicing；ResourceQuota+PriorityClass |
| 弹性触发 | 队列>10、GPU>85%、KV>80%、P99>2s；缩容 stabilization 600s、每次缩 1 |
| 容灾 | Xid>0 即隔离换卡；多 AZ<30s 切换；四级降级 L1 减 batch→L2 切小模型→L3 限流→L4 规则引擎 |
| PD 分离 | QPS>10 且有独立集群+资源差异大才上；ChunkFuse 1.3–1.5x 先试，DistServe 2x+ 需 RDMA |
| 可观测告警 | TTFT P99>0.8s warn / >2s crit；显存>95%；温度>85℃；Xid>0；队列>50；错误率>5% |
| GPU 利用率健康带 | 70–95%；<30% 浪费；>98% 将 OOM |
| 成本构成 | GPU 60–80%；单 1K token = GPU时费 ÷(单卡每小时 token÷1000) |
| 自建 vs 云 | <100万 token/月 API；100万–1亿 云 GPU/混合；>1亿 自建；月 API>$5K 评估自建 |
| 盈亏平衡 | 月>2亿 token 自建开始省钱；例 H100 $50K ÷($0.01−$0.002)/1K = 6.25B token 回本 |
| 容量公式 | GPU = ceil(峰值QPS×平均out tok ÷ 单卡吞吐) ÷ 利用率；7B≈5,500、70B≈550 tok/s(H100) |
| Buffer | 正常+30% / 峰值+50–100% / 容灾 N+1；组合 60%预留+25%按需+15%Spot |
| 优化累计 | 选模型→CB→量化→缓存→弹性，综合省 80–95% |
