> 由 SKILL.md 路由加载：可照抄的命令手册、排障表、工具链、术语、开源索引。
# 06 实操手册与资源索引

本模块为 FDE Skill 提供"可直接照抄的命令与步骤"和"不知道时该去查什么"。所有命令均来自原始实验文档，参数含义与典型取值已标注；不确定处标 [需确认]。

---

## 一、可照抄的实操手册

### 1.1 单机 7B 模型用 vLLM 起步部署

**环境要求**：GPU ≥ 24GB 显存（A100-80G / RTX 4090 等）；Python ≥ 3.10；CUDA ≥ 12.1。

**步骤**

```bash
# 1. 安装
pip install vllm

# 2. 单 GPU 启动（最简）
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000

# 3. 测试推理（OpenAI 兼容接口）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "你是一个助手"},
      {"role": "user", "content": "解释一下什么是 KV Cache？"}
    ],
    "max_tokens": 200
  }'

# 4. 实时观察显存
watch -n 1 nvidia-smi
```

**生产/压测推荐配置**

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.90 \   # 显存使用比例，留 10% 余量给波动
  --max-model-len 4096 \            # 最大序列长度，按业务调整
  --max-num-seqs 256 \              # 最大并发序列数，受 KV Cache 限制
  --enforce-eager false \           # 默认 False（开 Graph 优化更快）；调试时设 true
  --host 0.0.0.0 --port 8000
```

**验收方式**：① 启动无报错；② curl 返回正常推理结果；③ `nvidia-smi` 显存已占用；④ 有请求时 GPU 利用率 > 50%。

---

### 1.2 OOM 排查（按此顺序）

**标准现象**：`torch.cuda.OutOfMemoryError: CUDA out of memory.`

**排查顺序**

1. `nvidia-smi` —— 看当前显存占用与进程列表。
2. 分析显存构成：权重（FP16: 7B×2B≈14GB）+ KV Cache（取决于 batch/seq_len）+ 其他（1-2GB）。
3. `ps aux | grep vllm` —— 是否多进程加载了多个模型。
4. `fuser -v /dev/nvidia*` —— 是否有未释放显存的前进程。
5. 核对启动参数 `--max-model-len` / `--max-num-seqs` 是否过大。

**常见原因对照表**

| 原因 | 排查方法 | 解法 |
|------|----------|------|
| 多进程加载多个模型 | `ps aux \| grep vllm` | 确保只有一个进程 |
| 前进程未释放显存 | `fuser -v /dev/nvidia*` | `kill` 旧进程 |
| `max-model-len` 过大 | 检查启动参数 | 调小 |
| `max-num-seqs` 太大 | 检查启动参数 | 调小（如 256→128） |
| 其他进程占显存 | `nvidia-smi` 进程列表 | 清理无关进程 |

**清理 + 保守重启**

```bash
ps aux | grep vllm | grep -v grep | awk '{print $2}' | xargs kill -9
vllm serve model_name \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 128
```

**预防**：启动固定 `gpu-memory-utilization 0.90`；按模型大小合理配 `max-model-len`/`max-num-seqs`；加显存监控脚本。

---

### 1.3 性能 Profiling（看什么指标）

**工具链**

| 工具 | 层级 | 用途 |
|------|------|------|
| `nvidia-smi` | 系统级 | 实时监控 GPU 状态 |
| `nvtop` | 系统级 | 交互式 GPU 监控 |
| `nsys` (Nsight Systems) | 系统级 | CPU+GPU 时间线分析 |
| `ncu` (Nsight Compute) | Kernel 级 | 每个 kernel 性能详情 |

**步骤**

```bash
# 终端1：监控
watch -n 1 nvidia-smi
# 终端2：起服务
vllm serve Qwen/Qwen2.5-7B-Instruct
# 终端3：发请求观测
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"讲一个故事"}],"max_tokens":500}'

# 系统级 profiling（生成报告后看 CPU/GPU 时间占比、kernel 分布）
nsys profile --stats=true --sample=cpu --trace=cuda,nvtx \
  python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct
```

**瓶颈判断**

| 现象 | 类型 | 优化方向 |
|------|------|----------|
| GPU 利用率 > 80% | Compute-bound | Flash Attention、更大模型 |
| GPU 利用率 < 50%，带宽满 | Memory-bound | 量化、减少显存读写 |
| GPU 利用率 < 50%，带宽空闲 | IO-bound / batch 太小 | 批量传输、Pin Memory、增大并发 |

验收：有请求时 GPU 利用率 > 70% 为健康。

---

### 1.4 量化落地完整流程与验证

**目标**：7B 从 FP16 → INT8/INT4，验证精度与性能差异。

```bash
# 1. 安装量化工具
pip install autoawq vllm

# 2. 直接用预量化模型起服务（最省事）
vllm serve hugging-quants/Meta-Llama-3-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --host 0.0.0.0 --port 8000

# 3. 精度验证（跑基准评测对比 FP16 与量化版）
python -c "from lm_eval import evaluator; \
  # 分别跑 FP16 与 INT4 的 MMLU，对比分数差异 [需确认具体 task 配置]"

# 4. 性能压测
ab -n 100 -c 10 http://localhost:8000/v1/chat/completions
```

**FP16 vs INT4(AWQ) 对照**

| 指标 | FP16 | INT4 (AWQ) |
|------|------|------------|
| 权重显存 | 16 GB | 4 GB |
| 推理速度 | 基准 | 略慢（需反量化） |
| MMLU 分数 | 基准 | 下降 1-3% |
| 最大 batch | 受限 | 可大幅增大 |

**验收**：记录 P50/P99 延迟、Throughput (tok/s)；精度下降在可接受范围（一般 < 3%，超过 [需确认阈值] 需回退或换量化方案）。INT4 不一定更快（反量化开销），仅当显存受限、需扩大 batch 时使用。

---

### 1.5 Batching / 并发调优实验方法

**基准测试（vLLM 自带 benchmark）**

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.90 --max-num-seqs 256 --max-model-len 4096

python -m vllm.entrypoints.openai.benchmark \
  --backend vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset-name random \
  --random-input 1024 --random-output 128 \
  --max-num-seqs 256
```

**`max-num-seqs` 影响（典型趋势）**

| max-num-seqs | 吞吐 (tok/s) | P99 延迟 | 显存利用率 |
|--------------|-------------|----------|-----------|
| 32 | 基准 | 最低 | 低 |
| 128 | ~2x | 略高 | 中 |
| 256 | ~3x | 较高 | 高 |
| 512 | 可能 OOM | 很高 | 极高 |

**`gpu-memory-utilization` 取值**

| 取值 | 效果 |
|------|------|
| 0.80 | 留 20% 余量，安全但浪费 |
| 0.90 | 推荐，平衡安全与性能 |
| 0.95 | 激进，可能偶发 OOM |

**结论**：吞吐随 batch 增加但非线性；延迟同步上升；最优值在业务的 QPS 与延迟要求（如 P99 < 500ms）间取平衡。短回复（<50 token）可激进增大 batch，长回复（>500 token）需保守。

---

### 1.6 张量并行（TP）实验

**目标**：4×A100-80G（NVLink）上用 TP 部署 70B。

**显存估算先行**

```
70B FP16：权重 70×2=140GB；TP=4 时每卡 35GB；KV Cache(batch=32,seq=4096) 每卡~8GB → 每卡~43GB，远低于 80GB，可行。
TP=2：每卡权重 70GB + KV~16GB = ~86GB → OOM，不可行。
```

**启动**

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 4 \
  --quantization fp8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

**验证**：`nvidia-smi` 应显示 4 张卡显存占用基本相同；curl 测试返回正常。

**性能对比（典型）**：TP=4 为基准；TP=8 吞吐约为 4 卡的 60%、延迟约 1.5x，通信开销占比随 TP 增大而升高、scaling 效率下降。跨服务器 TP=8 不推荐（通信代价高），优先 TP=4 + DP=2（两组 TP=4 副本）。[需确认具体模型与互联条件]

---

## 二、工具链清单

| 环节 | 工具 | 一句话用途 |
|------|------|-----------|
| 推理部署 | vLLM | 高并发开源推理引擎，PagedAttention + Continuous Batching |
| 推理部署 | SGLang | 结构化生成（JSON/函数调用/多轮）优化，RadixAttention |
| 推理部署 | llama.cpp | 跨平台纯 C/C++ 推理，CPU/GPU 混合，GGUF 量化事实标准 |
| 推理部署 | TensorRT-LLM | NVIDIA 官方推理优化框架 |
| 量化 | autoawq / GPTQ | 训练后权重量化（AWQ/GPTQ） |
| 压测 | vLLM benchmark | 自带吞吐/延迟基准测试 |
| 压测 | ab / wrk | HTTP 层并发压测 |
| 系统监控 | nvidia-smi / nvtop | 实时 GPU 显存与利用率 |
| 性能剖析 | nsys (Nsight Systems) | CPU+GPU 时间线、系统级瓶颈定位 |
| 性能剖析 | ncu (Nsight Compute) | 单 kernel 级性能详情 |
| 精度评测 | lm_eval (EleutherAI) | 标准化基准（MMLU 等）精度对比 |
| AI 编程（编辑器） | Cursor | 图形 AI 编辑器，Tab/Cmd+K/Agent Mode |
| AI 编程（终端） | Claude Code | 终端 Agent，批量操作/DevOps/脚本 |
| 规范驱动 | OpenSpec | Spec-first 开发框架，单一信息源 |
| 护栏/质量 | Harness (C3 不变量) | 编译/门禁/边界/不可停守卫 |
| 集成协议 | MCP / A2A | 模型与外部系统/其他 Agent 通讯 |

---

## 三、高频术语表

**模型架构**：Transformer（Self-Attention 序列模型）；KV Cache（缓存 Decode 阶段 K/V，避免重复计算）；GQA/MQA（多 query 共享 KV，省显存）；RoPE（旋转位置编码）；MoE（每 token 经 Router 选部分专家）；MLA（DeepSeek 的 Attention 变体）。

**推理引擎**：PagedAttention（vLLM 分页管理 KV Cache）；Continuous Batching（动态调度，不等最长请求）；In-flight Batching（TRT-LLM 更细粒度）；TTFT（首 token 延迟）；TPOT（每 token 生成时间）；Throughput（tokens/s/GPU）。

**量化**：PTQ（训练后量化）；QAT（量化感知训练）；AWQ（保护大激活对应权重）；GPTQ（逐层量化考虑误差累积）；SmoothQuant（平滑激活与权重）；FP8/INT8/INT4（8/8/4-bit 格式，INT4 极致压缩）。

**GPU/底层**：SM（GPU 基本执行单元）；Tensor Core（矩阵乘专用）；HBM（高带宽显存）；NVLink（GPU 间 600GB/s 互联）；TP（层内矩阵切分）/PP（层间切分）/DP（请求分发副本）/MIG（单卡切多实例）。

**部署运维**：SLO（服务等级目标）；HPA（K8s 水平扩缩）；Canary（灰度放量）；RAG（检索增强生成）；Agent（能调工具、自主规划的智能体）；Speculative Decoding（小模型预测加速大模型）；Flash Attention（省显存的高效 Attention）。

---

## 四、开源项目索引（docs-opensource，共 38+ 篇）

> 索引级浏览：下面对应 39 篇源码解读，按"解决什么问题 / 什么时候去看"组织。Claude Code 部分按 10 大模块分组（每组含 1-6 篇）。

| 项目 / 模块 | 解决什么问题 | 什么时候去看 |
|------|------|------|
| **nanoGPT** | 最精简 GPT 实现（~300 行），教学级理解 Transformer 组件如何代码化 | 想搞懂 Attention 维度变换、训练/推理循环时 |
| **llm.c** | 纯 C 实现 GPT-2 推理/训练，看 Python→C 的物理映射 | 想理解推理引擎底层计算、CUDA 优化时 |
| **llama.cpp** | 跨平台纯 C/C++ 推理，GGUF 量化标准，CPU/GPU 混合 | 消费级硬件/端侧/量化落地、无 GPU 推理时 |
| **vLLM** | PagedAttention + Continuous Batching，云端高并发吞吐 | 做生产级高吞吐推理服务、排查显存碎片/OOM 时 |
| **SGLang** | RadixAttention + FSM 约束生成，结构化/函数调用/多轮性能 | 做 Agent、JSON 约束输出、多轮对话复用前缀时 |
| **Claude Code 总览** (00-文档导航) | 51 万行 TS 源码的整体目录映射与学习路径 | 要系统学 Agent CLI 架构、规划阅读顺序时 |
| **CC 01 启动流程** (应用入口/初始化状态) | CLI 启动、bootstrap、配置初始化 | 排查启动失败、理解入口流程时 |
| **CC 02 Prompt 系统** (规范/动态/上下文注入) | system prompt 分层、动态生成、上下文注入 | 设计 Agent 的 prompt 结构、规则层时 |
| **CC 03 核心引擎** (查询编排/引擎循环/Token 预算) | 查询编排、引擎主循环、token 预算管理 | 理解 Agent 如何调度工具、控成本时 |
| **CC 04 工具系统** (基类/注册/Bash/文件/Agent/权限) | 45 个工具的定义、注册、权限模型 | 自研工具、设计工具权限与沙箱时 |
| **CC 05 记忆系统** (概述/写入/读取) | 跨会话 .md 记忆的写入与读取 | 设计 Memory/RAG、知识沉淀时 |
| **CC 06 状态管理** (状态存储) | 会话状态外部化存储 | 长任务状态机、断点恢复时 |
| **CC 07 UI 渲染** (Ink/组件/Hooks) | 终端 UI（Ink）渲染与 Hooks | 做 CLI 交互界面时 |
| **CC 08 服务集成** (MCP/LSP/OAuth) | MCP 协议、LSP、OAuth 集成 | 接入外部服务/IDE/鉴权时 |
| **CC 09 扩展系统** (Skills/Plugins/命令) | Skill、插件、斜杠命令扩展机制 | 封装可复用 Skill、扩展能力时 |
| **CC 10 安全系统** (Bash安全/权限/路径/策略) | Bash 安全、权限控制、路径验证、策略限制 | 设计 Agent 安全边界、防越权时 |
| **CC 记忆系统对比** (70-) | Claude Code vs Copilot vs Cursor 记忆对比 | 选型或设计记忆机制时 |

---

## 五、AI 工程实践铁律（来自 blog 9 篇，对 Skill 行为准则至关重要）

以下规则决定了"让 AI/Agent 稳定干活"的硬性纪律，是 Skill 自身应遵循的行为准则。

### 5.1 思考前置（Think Before Coding）
动手前先分析原始需求、真实目标、客观约束、潜在风险。目标不清晰先停下来与用户确认；路径非最短/最简/最稳直接指出；若用户把表面症状当核心问题，继续追问。**不可跳过的环节：约束澄清必须有人参与交互，不能交给 AI 自问自答。**

### 5.2 约束前置且必须可执行（机器可验证）
凡未明说的部分，Agent 会用训练数据"通用模式"自行合理化填充——这是最危险的。约束须包含：描述、约束等级（硬卡点/软卡点/建议）、验证方式、验证阶段、预期结果。**铁标准：不能被机械化验证的约束等于无效约束**（文档写再漂亮没人逐条核对 = 没写）。每个任务只绑定相关约束子集，避免信息过载。

### 5.3 流程编排六阶段（职责分离）
1. **只读不写**：先读代码库、做架构考古/Git 热点/依赖追踪，禁止写代码。
2. **想不做**：列可选方案、对比优劣，不写代码。
3. **规范文档**：设计 + 任务拆解 + 验收标准，把"我觉得"变"你需要做这三件、验收四个条件"。
4. **按规范实施**：TDD，逐任务落地。
5. **规范对齐验证**：逐条核对约束清单是否被遵守。
6. **安全审查**：攻击面、数据流、依赖漏洞扫描。

流程可裁剪（熟项目可跳侦察，简单需求可并探索），核心价值在职责分离，不在全跑 checklist。变更需反向同步到方案文档，保持方案与代码一致。

### 5.4 验收闭环三道关
- **入口关（编码前）**：硬卡点为零、约束完备才许进入编码。
- **评审关（发布前）**：硬卡点风险为零、约束覆盖率达标才许提交。
- **上线关（发布前）**：影响范围可控、风险评级 ≤ 中才许上线。

重点检测 AI 特有缺陷：调用 API 是否真实存在（幻觉 API）、是否过度实现、约束是否遗忘、与设计偏差是否超阈值。**机器守硬卡点，人看软约束；机器查范围，人判方向。**

### 5.5 上下文管理：三级 + 三阶段压缩
上下文不是越多越好——窗口越大越易迷失重点。
- **三级**：① 必传不可压缩（约束清单/硬卡点/资金安全规则）；② 可摘要压缩（方案核心/变更摘要）；③ 参考性按需注入（知识库/历史）。
- **三阶段压缩**：结构化提取骨架 → 摘要压缩结论/数据 → 紧急截断第三级、精简第二级。**原则：约束清单任何阶段都不可压缩，是硬红线。**
- 知识防腐烂：PR 关联知识更新强制确认、定时巡检、AI 起草+人审签。

### 5.6 子智能体隔离上下文（防上下文炸裂）
每个任务用独立子智能体执行，主会话只做调度。避免单 session 塞入过多不相关信息导致混淆不同任务的约束与状态。单 Agent 持有 > 15 个工具时调用正确率显著下降——按 context 边界切分 Agent，而非按问题类型。

### 5.7 TDD 是默认而非可选
红-绿-重构为铁律，杜绝"事后补测试"。测试不仅是 QA，更是"AI 理解对不对"的即时反馈。设计/计划/代码三者强一致、可互相追溯。

### 5.8 Harness 四层架构（连接/编排/效能/安全）
- **连接**：用 Tool Calling / MCP / A2A 把外部能力包装成模型可通讯对象。
- **编排**：用外部状态机记录任务边界、过程状态、下一步选择、异常恢复、结果验收，不依赖模型"记忆"。
- **效能**：用确定性复用替代重复推理——上下文管理、Memory/RAG、缓存、模型路由、Skill、规则脚本；能 CPU 确定性执行的就不每次让 GPU 大模型推理。
- **安全**：见 5.9。

关键张力：效能求复用 vs 安全求隔离；编排求自主推进 vs 安全求关键节点停等人。好的 Harness 是在张力中取平衡。

### 5.9 安全四层防线（不可交给模型自己）
1. **输入侧防注入**：区分指令与数据，标记上下文可信等级，防外部文本改变 Agent 行为。
2. **输出侧结构化约束**：Schema/parser/validator/guardrails，输出必须通过确定性检查才能进入下一步。
3. **执行侧权限与沙箱**：Auth/Policy/Sandbox/Quota——能读不能写、能查个人不能查全量、能跑脚本不能联网。
4. **高风险动作人在回路 + 全链路审计**：付款/删除/外发/权限变更/生产发布须审批点；Audit Log 是进入生产的前提，不是事后补丁。

### 5.10 不变量守卫（C3）优于审批节点
定义硬性不变量，范围内给 AI 完全自由（这是 Harness 与 Workflow 的本质区别）：
- `guard-compile-required`：改代码必须编译通过。
- `guard-gate-check`：进下一阶段须满足前置，不可跳步。
- `guard-path-boundary`：AI 只能访问声明的工作空间。
- `guard-stop`：未完成必要阶段不许停止。
- `guard-evolution`：Harness 自我修改每会话上限 2 次，防无限漂移。

### 5.11 人类兜底责任（Human-in-the-Loop 不可跳过）
AI 写错代码导致线上故障，责任在人类工程师。无论 AI 多靠谱，编排者必须理解 AI 做了什么、为什么、出事怎么修。审查从"泛泛看"变"聚焦看"；**生产环境永不跳过审批**（Claude Code 等工具的 dangerously-skip-approval 仅限 CI/CD）。AI 擅长执行与试错，真正跳出框架的创新性 idea、方向定义、最终兜底仍属人类。

---

*编写说明：命令与参数以各工具官方文档为准，落地前必须按现场环境实测；带 [需确认] 处为需现场核实项。*
