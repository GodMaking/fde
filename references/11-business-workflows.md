> 由 SKILL.md 路由加载：业务流程编排、企业系统集成、业务指标体系——把技术成果翻译成业务价值。

# 业务流程、系统集成与业务指标

## 一、从「模型部署」到「业务价值」

### 1.1 两个视角的错位

模型部署完成 ≠ 业务成功。两边看的是完全不同的东西：

```
技术团队视角：  GPU 利用率 85% | P99 延迟 500ms | Token 成本降低 40%   → 完成
业务团队视角：  工单处理时长没变 | 客户满意度没提升 | 人力成本没下降   → 没价值
```

业务价值实现的完整闭环：

```
模型部署完成（技术指标达标）
  → 识别业务场景（哪个流程最需要 AI？）
  → 设计工作流（AI 在流程中扮演什么角色？）
  → 集成企业系统（CRM/ERP/API 对接）
  → 设计 Human-in-the-Loop 审核机制
  → 小范围试点上线
  → 指标监控（技术指标 + 业务指标）
  → 业务指标达标？──否→ 分析瓶颈、调整流程/模型 → 回到「设计工作流」
                  └─是→ 全量推广 → 持续优化（A/B 测试 + Drift 检测）→ 回到「指标监控」
```

### 1.2 AI 在商业流程中的 5 种角色

| 角色 | 定义 | 典型场景 | AI 自主度 | 人工介入 |
|------|------|---------|----------|---------|
| 替代 Replace | AI 完全替代人工操作 | 自动回复 FAQ、智能路由 | 高 | 无需 |
| 辅助 Assist | AI 提供信息和建议，人工决策 | 客服辅助推荐回复 | 中 | 必须 |
| 增强 Augment | AI 增强人工能力，提高效率 | 工单自动摘要、文档翻译 | 中低 | 审核 |
| 决策 Decide | AI 自动做出业务决策 | 信用评分、风险预警 | 高 | 异常时介入 |
| 编排 Orchestrate | AI 协调整个工作流 | 多步骤工单处理、跨系统操作 | 中高 | 关键节点审核 |

技术要求与风险由低到高：辅助 → 增强 → 决策 → 编排 → 替代。落地路径从「辅助」开始，逐步向「增强」「编排」推进，最后才考虑「替代」。

### 1.3 商业工作流与传统自动化的区别

```
传统自动化（RPA / 规则引擎）：
  IF 工单包含"退款" THEN 转退款组
  IF 工单包含"投诉" THEN 优先级设为高

AI 商业工作流：
  理解工单语义 → 提取关键信息 → 查询客户历史 → 生成个性化回复 → 人工审核 → 发送
```

### 1.4 六个核心概念

| 概念 | 说明 | 在体系中的位置 |
|------|------|---------------|
| Workflow（工作流） | 定义好的步骤序列，有明确输入输出 | 业务层 |
| Orchestration（编排） | 协调多个服务、工具、模型的执行顺序 | 中间层 |
| Human-in-the-Loop（人在回路） | 关键节点引入人工审核和决策 | 安全层 |
| State Machine（状态机） | 用状态转移描述流程的每一步 | 实现层 |
| Compensation（补偿） | 流程失败时的回退和恢复机制 | 可靠性层 |
| Observability（可观测性） | 业务指标和技术指标的映射 | 运营层 |

---

## 二、业务流程编排

### 2.1 工作流 vs Agent

| 维度 | Agent | 业务工作流 Workflow |
|------|-------|---------------------|
| 自主性 | 高：自己决定下一步 | 低：流程预先定义 |
| 灵活性 | 动态规划，可调整策略 | 固定步骤，可配置分支 |
| 可控性 | 低：可能出现意外行为 | 高：每一步都可审计 |
| 适用场景 | 探索性任务（研究、分析） | 确定性流程（工单处理、审批） |
| 可预测性 | 每次执行路径可能不同 | 相同输入产生相同流程 |
| 调试难度 | 高：非确定性行为 | 低：日志可追踪每一步 |
| 合规要求 | 难满足（黑盒决策） | 易满足（每步可审计） |

在商业场景中，**工作流是主流，Agent 是补充**。

### 2.2 典型工作流：客户邮件自动处理

```
收到客户邮件
  → AI 自动分类（工单类型识别：退款/投诉/咨询/技术支持）
  → 置信度 > 0.9 ?──是→ 调取客户资料（CRM 查询）
                 └─否→ 切人工分类（标注并学习）→ 也进入调取资料
  → 生成回复草稿（LLM + 模板）
  → 人工审核（客服确认/修改）
  → 发送邮件 → 归档记录（更新工单状态）
  → 反馈循环（记录人工修改 → 微调模型）
```

关键设计点：自动分类用 NLP 模型识别类型；置信度阈值 0.9；AI 回复须人工确认才能发送；人工修改过的回复回流用于微调，形成数据飞轮。

### 2.3 Human-in-the-Loop 三种模式

**模式 1：前置审核（Pre-Approval）**

```
人工确认输入 → AI 生成 → 自动执行
```

适用：输入质量不确定、AI 易误解意图；生成成本低但执行成本高（如发邮件给客户）。典型：客服先确认客户问题，AI 再检索知识库生成回复。
优点：避免基于错误理解做错误操作；缺点：增加人工步骤，降低自动化率。

**模式 2：后置审核（Post-Approval）**

```
AI 生成 → 人工审核 → 通过则执行 / 不通过则修改或拒绝
```

适用：AI 生成质量较高（>90% 通过率）；执行后果较重（法律文件、对外通信）。典型：AI 生成客服回复草稿，客服确认后发送。
优点：AI 做大部分工作，效率提升明显；缺点：通过率低时人工审核成为瓶颈。

**模式 3：抽检审核（Spot-Check）**

```
AI 生成 → 自动执行 → 定期随机抽样（5%-10%）人工审核 → 合格率 > 95% ? 继续自动运行 : 切全量审核 + 触发模型评估
```

适用：AI 已稳定运行较长时间、质量可信；执行后果较轻（内部报告、分类标记）。典型：AI 自动分类工单，每周抽检 10% 验证准确率。
优点：自动化率最高、人工成本最低；缺点：可能漏掉错误，发现问题延迟较高。

**三种模式对比**

| 维度 | 前置审核 | 后置审核 | 抽检审核 |
|------|---------|---------|---------|
| 自动化率 | 30-50% | 60-80% | 95%+ |
| 安全性 | 最高 | 高 | 中 |
| 人工成本 | 最高 | 中 | 最低 |
| 适用阶段 | 新系统上线 | 系统成熟期 | 系统稳定期 |
| 切换条件 | AI 分类准确率 > 85% 后切换 | AI 审核通过率 > 90% 后切换 | 连续 4 周合格率 > 95% |

### 2.4 异常分支与降级

异常分类与对应处理：

| 异常类型 | 触发条件 | 处理动作 |
|---------|---------|---------|
| AI 超时 | > 30s | 降级处理：返回预设回复 + 标记人工跟进 |
| 置信度低 | < 0.7 | 切人工，附带 AI 建议作为参考 |
| 系统故障 | CRM 不可用等 | 进入重试队列，指数退避重试 |
| 业务规则冲突 | 客户黑名单等 | 直接转人工，跳过 AI 处理 |

所有异常分支最终汇入「异常归档 → 事后分析」。

超时降级的两级策略：

```python
TIMEOUT_MS = 30_000  # 30 秒超时

async def handle_request_with_fallback(request):
    try:
        result = await asyncio.wait_for(ai_process(request), timeout=TIMEOUT_MS / 1000)
        return result
    except asyncio.TimeoutError:
        # 降级 1：返回预设兜底回复
        metrics.record("ai_timeout")
        return {"type": "fallback",
                "message": "我们正在处理您的请求，请稍候...",
                "ticket_id": create_ticket(request)}
    except ExternalServiceError as e:
        # 降级 2：外部服务不可用，切人工
        metrics.record("external_service_error")
        return route_to_human(request, reason=str(e))
```

**基于置信度的路由（Confidence-Based Routing）**

```
置信度 >= 0.9   → AI 自动处理
置信度 0.7~0.9  → AI 生成 + 人工审核
置信度 < 0.7    → 直接转人工 + AI 建议作为参考
```

### 2.5 状态机建模

工单处理状态流转：

```
[*] → NEW（收到工单）
NEW → AI_CLASSIFYING（开始分类）
AI_CLASSIFYING → CLASSIFIED（分类完成）
AI_CLASSIFYING → ESCALATED（置信度低）

CLASSIFIED → AI_PROCESSING（自动处理）
CLASSIFIED → HUMAN_REVIEW（需要人工审核）

AI_PROCESSING → COMPLETED（处理成功）
AI_PROCESSING → RETRYING（处理失败）
AI_PROCESSING → ESCALATED（多次失败）

RETRYING → AI_PROCESSING（重试成功）
RETRYING → ESCALATED（超过最大重试次数）

HUMAN_REVIEW → AI_PROCESSING（审核通过）
HUMAN_REVIEW → MODIFIED（人工修改）
HUMAN_REVIEW → REJECTED（审核驳回）

MODIFIED → COMPLETED（修改后发送）
ESCALATED → HUMAN_REVIEW（人工接手）
COMPLETED → [*]
ESCALATED → [*]（人工处理完成）
```

状态机设计四原则：

| 原则 | 说明 | 示例 |
|------|------|------|
| 每个状态必须有出口 | 不能有死状态 | RETRYING 必须能到 ESCALATED |
| 超时转移 | 每个状态设超时 | AI_CLASSIFYING 30s 超时 → ESCALATED |
| 幂等性 | 同一操作重复执行不产生副作用 | 重复工单创建返回已有 ticket_id |
| 可审计 | 每次状态转移记录日志 | NEW → AI_CLASSIFYING 记录时间戳、原因 |

### 2.6 工作流引擎选型

| 工具 | 定位 | LLM 集成 | 学习曲线 | 适用场景 |
|------|------|---------|---------|---------|
| Temporal | 代码优先的分布式工作流 | 原生支持 Python/Go SDK 调用 LLM | 中 | 复杂编排、长期运行任务 |
| Camunda | BPMN 标准的工作流引擎 | 通过 REST API 调用 LLM | 高 | 企业级审批流、合规要求高 |
| Airflow | 数据管道编排 | 通过 PythonOperator 调用 LLM | 低 | 批处理、ETL + LLM |
| LangGraph | Agent 工作流框架 | 原生 LLM 集成 | 低 | Agent 场景、状态机驱动的 Agent |
| Dify | 低代码 AI 应用平台 | 内置 LLM 节点 | 极低 | 快速搭建 AI 工作流原型 |

Temporal 编排 AI 工单的骨架（Activity + Workflow）：

```python
@activity.defn
async def classify_ticket(ticket: dict) -> dict: ...      # 调 LLM 分类，返回 category + confidence

@workflow.defn
class TicketProcessingWorkflow:
    @workflow.run
    async def run(self, ticket: dict) -> str:
        classification = await workflow.execute_activity(
            classify_ticket, ticket, start_to_close_timeout=timedelta(seconds=30))
        if classification["confidence"] < 0.7:
            await workflow.execute_activity(escalate_to_human, ticket)
            return "escalated"
        customer_data = await workflow.execute_activity(fetch_customer, ticket["customer_id"])
        reply = await workflow.execute_activity(generate_reply, ticket, customer_data)
        approved = await workflow.execute_activity(wait_for_human_approval, reply)  # HITL
        if not approved:
            return "rejected"
        await workflow.execute_activity(send_email, ticket["email"], "回复", reply)
        return "completed"
```

Worker 水平扩展用 HPA，以队列深度为指标：`temporal_task_queue_depth` 平均值达到 100 即扩容，minReplicas=3、maxReplicas=20。

部署模式三选：Worker Deployment（多 Pod 水平扩展，高并发工单处理）、Sidecar Pattern（与业务服务共部署，轻量级）、External SaaS（Temporal Cloud / Camunda Cloud，免运维）。

---

## 三、企业系统集成

### 3.1 需要对接的系统全景

| 系统类型 | 代表产品 | AI 读取 | AI 写入 | 典型场景 |
|---------|---------|--------|--------|---------|
| CRM | Salesforce、HubSpot | 客户画像、工单历史、购买记录 | 创建工单、更新状态、添加备注 | 智能客服、销售辅助 |
| ERP | SAP、Oracle | 订单状态、库存数量、财务数据 | 创建订单、更新库存 | 订单自动处理 |
| 内部 API | 自研系统 | 权限信息、审批状态、计费数据 | 发起审批、更新权限 | 内部流程自动化 |
| 消息平台 | Slack、飞书、企业微信 | 消息内容、群组信息 | 发送通知、创建频道 | 智能通知、告警 |
| 数据库 | PostgreSQL、MySQL | 业务状态查询、数据聚合 | 更新记录、插入日志 | 数据分析、报表 |

集成复杂度三档：

```
低复杂度：只读 + REST API + 同步调用   → 知识库检索、信息查询
中复杂度：读写 + REST API + 异步调用   → 工单创建、状态更新
高复杂度：读写 + 多种协议 + 事务保证   → 订单处理、库存扣减
```

### 3.2 集成架构

```
AI 服务层：LLM 推理服务（vLLM / API）→ Agent/工作流引擎（Temporal / LangGraph）→ 工具调用层（Tool Registry）
集成层：  API Gateway（Kong / APISIX）、消息队列（Kafka / RabbitMQ）、Webhook Server
业务系统：CRM(Salesforce)、ERP(SAP)、业务数据库(PostgreSQL)、消息平台(Slack/飞书)、认证服务(OAuth2/SSO)
```

三条原则：AI 服务不直接调用业务系统 API，统一经 API Gateway 路由与管理；事件通过消息队列异步传递实现解耦；Webhook Server 接收外部系统事件通知。

### 3.3 Webhook 与事件驱动

轮询 vs 事件驱动：

| 对比维度 | 轮询 Polling | 事件驱动 Webhook |
|---------|-------------|-----------------|
| 延迟 | 高（通常 1-5 分钟） | 低（实时推送，秒级） |
| 资源消耗 | 高（无效请求多） | 低（仅事件时触发） |
| 实现复杂度 | 低（定时任务） | 中（需签名验证、重试） |
| 可靠性 | 中（漏轮询会丢失） | 高（可配合消息队列保证） |
| 适用场景 | 低频数据同步 | 实时响应场景 |

推荐：核心流程用 Webhook 事件驱动，非关键数据用定时轮询兜底。

**签名验证**（必做，HMAC-SHA256 + 时间安全比较防时序攻击）：

```python
def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)

@app.post("/webhook/salesforce")
async def handle_salesforce_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Salesforce-Signature", "")
    if not verify_webhook_signature(body, signature, WEBHOOK_SECRET):
        raise HTTPException(status_code=401, detail="Invalid signature")
    event = parse_cloud_event(body)
    await process_event(event)
    return {"status": "ok"}
```

**事件格式统一用 CloudEvents**：

```json
{
  "specversion": "1.0",
  "type": "com.salesforce.ticket.updated",
  "source": "/crm/salesforce",
  "id": "evt-12345-abcde",
  "time": "2025-06-15T10:30:00Z",
  "data": {"ticket_id": "TKT-7890", "status": "escalated",
           "customer_id": "CUST-123", "assigned_to": "agent-456"}
}
```

| 字段 | 必填 | 说明 |
|------|------|------|
| specversion | 是 | CloudEvents 版本 |
| type | 是 | 事件类型，反向域名格式 |
| source | 是 | 事件来源 |
| id | 是 | 全局唯一事件 ID（用于去重） |
| time | 是 | 事件发生时间（ISO 8601） |
| data | 否 | 事件载荷 |

### 3.4 数据同步与一致性

| 策略 | 延迟 | 可靠性 | 复杂度 | 适用场景 |
|------|------|--------|--------|---------|
| 强一致性 | 高（同步等待） | 最高 | 低 | 支付、库存扣减 |
| 最终一致性 | 低（异步处理） | 中 | 中 | 客户信息同步、状态更新 |
| 混合模式 | 中 | 高 | 高 | 关键数据强一致，非关键最终一致 |

强一致：写入 CRM → 等待确认 → 更新本地缓存。最终一致：写入 CRM → 发送事件到 MQ → 异步消费更新。

**CDC（Change Data Capture）**，不侵入业务代码：

```
数据库 Binlog → CDC（Debezium）→ Kafka → AI 服务消费事件
                                    └→ 审计日志（数据溯源）
```

优势：不侵入业务代码，监听数据库层变更；保证变更事件的顺序与至少一次送达；可同时服务多个消费者（AI 服务、审计日志、报表）。

**数据版本控制（乐观锁）**，避免基于过期数据决策：

```python
def update_customer(customer_id: str, data: dict, expected_version: int) -> bool:
    current = db.get_customer(customer_id)
    if current.version != expected_version:
        raise ConflictError(f"Version mismatch: expected {expected_version}, got {current.version}")
    current.update(data); current.version += 1; db.save(current)
    return True
```

### 3.5 失败重试与补偿

**指数退避 + 随机抖动**（防惊群）：

```python
async def exponential_backoff_retry(func, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return await func()
        except (TimeoutError, ConnectionError) as e:
            last_error = e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)
    raise last_error
```

重试策略矩阵：

| 错误类型 | 是否重试 | 策略 |
|---------|---------|------|
| 网络超时 | 是 | 指数退避，最多 3 次 |
| 4xx 客户端错误 | 否 | 记录日志，返回错误 |
| 429 限流 | 是 | 等待 Retry-After 头指定的时间 |
| 5xx 服务端错误 | 是 | 指数退避，最多 3 次 |
| 业务逻辑错误 | 否 | 转入人工处理 |

**Saga 补偿事务**（跨系统最终一致）：

```
Step 1 在 CRM 创建工单 → Step 2 在 ERP 检查库存
  库存不足 → 补偿：删除 CRM 工单
  库存充足 → Step 3 在支付系统扣款
    扣款失败 → 补偿：恢复库存 + 删除工单
    扣款成功 → 完成
```

```python
class TicketSaga:
    async def execute(self, ticket_data: dict) -> str:
        try:
            ticket_id = await self.crm.create_ticket(ticket_data)
            stock = await self.erp.check_stock(ticket_data["product_id"])
            if not stock.sufficient:
                await self.compensate(ticket_id)
                raise InsufficientStockError()
            payment_id = await self.payment.charge(ticket_data["amount"])
            return ticket_id
        except Exception:
            await self.compensate(ticket_id)
            raise

    async def compensate(self, ticket_id: str):
        await self.crm.cancel_ticket(ticket_id)   # 按正向操作逆序补偿；有库存操作则恢复库存，有支付则退款
```

**死信队列（DLQ）与人工介入**：

```
正常请求 → 处理成功 → 返回结果
             ↓失败
         重试队列 → 重试成功 → 返回结果
             ↓超过最大重试次数
         死信队列（DLQ）→ 人工介入
```

DLQ 消息必须含完整上下文：

```json
{"original_message": {...}, "error": "Salesforce API returned 500",
 "retry_count": 3, "first_attempt": "2025-06-15T10:00:00Z",
 "last_attempt": "2025-06-15T10:07:00Z",
 "context": {"ticket_id": "TKT-7890", "customer_id": "CUST-123"}}
```

### 3.6 认证与限流

| 认证方式 | 安全性 | 复杂度 | 适用场景 |
|---------|--------|--------|---------|
| API Key | 低 | 极低 | 内部服务间调用 |
| OAuth2 Client Credentials | 中 | 中 | 服务间认证 |
| OAuth2 Authorization Code | 高 | 高 | 用户授权的 API 调用 |
| mTLS（双向 TLS） | 极高 | 高 | 金融、医疗等高安全场景 |
| JWT（Bearer Token） | 中 | 低 | 无状态认证 |

API Gateway 限流配置（Kong）：

```yaml
plugins:
- name: rate-limiting
  config: {hour: 10000, policy: redis, limit_by: consumer}   # 每小时 1 万次，Redis 分布式计数，按消费者限流
- name: request-size-limiting
  config: {allowed_payload_size: 1}                          # 最大 1MB 请求体
- name: correlation-id
  config: {header_name: X-Request-ID}                        # 全链路追踪 ID
```

集成层两种部署模式：

| 模式 | 优点 | 缺点 | 推荐场景 |
|------|------|------|---------|
| API Gateway | 集中管理、统一限流、统一认证 | 单点瓶颈、运维复杂 | 对外 API、多服务共享 |
| Sidecar（Service Mesh） | 透明代理、服务间 mTLS、细粒度控制 | 每 Pod 一个 Sidecar，资源开销 | 微服务架构、内部服务通信 |

### 3.7 集成最佳实践

1. 统一事件格式：外部事件统一用 CloudEvents。
2. Webhook 必验签：不验签等于公开 API，必须用 HMAC 签名。
3. 幂等性设计：所有写操作幂等，用业务 ID 做去重键，防重试导致数据重复。
4. 超时和限流：每个外部 API 调用设超时（建议 30s），客户端限流防打垮下游。
5. 补偿先行：设计跨系统操作时先想好怎么回退，再想正向执行。
6. 本地缓存兜底：关键数据（客户画像、产品目录）本地缓存，外部不可用时降级读缓存。
7. 全链路追踪：每个请求带 X-Request-ID，贯穿 AI 服务 → Gateway → 业务系统。
8. 定期健康检查：定时探测所有外部系统 API 可用性。

---

## 四、业务指标体系

### 4.1 技术指标 → 业务指标映射

**核心原则：每个技术指标都必须能映射到至少一个业务指标。**

| 技术指标 | 业务指标 | 映射方法 |
|---------|---------|---------|
| Token 生成速度 | 工单平均处理时长 | 统计响应时间与处理时长的相关性 |
| 模型准确率 | 客户满意度 / NPS | A/B 测试对比 |
| GPU 利用率 | 单次业务调用成本 | 成本分摊计算 |
| 并发数 | 峰值时段服务客户数 | 容量规划 |
| 幻觉率 | 工单升级率（Escalation Rate） | 人工介入比例统计 |
| 工具调用成功率 | 自动化流程完成率 | 端到端成功率统计 |

### 4.2 三层指标架构

```
技术层：GPU 利用率 85% | P99 延迟 800ms | 模型准确率 94% | Token 成本 $0.5/千次
   ↓ 映射
业务层：自动处理率 78% | 平均处理时长 3.2min | 客户满意度 4.5/5 | 单次处理成本 $0.12
   ↓ 映射
财务层：月度人力节省 ¥450K | 客户流失率降低 8% | ROI 340%
```

具体映射关系：GPU 利用率 → 单次处理成本；P99 延迟 → 平均处理时长；模型准确率 → 自动处理率 + 客户满意度；Token 成本 → 单次处理成本。业务层再映射：自动处理率 → 月度人力节省；平均处理时长 → 客户流失率；单次处理成本 → ROI。

**映射公式：**

```
单次处理成本 = (GPU 小时成本 / 每小时处理工单数) + API 调用成本
月度人力节省 = (AI 处理工单数 × 人工平均处理时间 × 人工时薪) - AI 运营成本
ROI = (月度人力节省 + 额外收入 - AI 总成本) / AI 总成本 × 100%
```

### 4.3 用 OKR 定义成功指标

```
Objective: 用 AI 将客服工单处理效率提升 50%

Key Results:
  KR1: 工单自动处理率从 20% 提升到 60%（业务指标）
  KR2: 平均工单处理时长从 15 分钟降到 8 分钟（业务指标）
  KR3: AI 处理工单的客户满意度 ≥ 人工处理的 90%（质量指标）
  KR4: 单次工单处理成本降低 40%（成本指标）
  KR5: 模型幻觉率 < 3%（技术指标，支撑 KR3）
```

制定注意事项：

1. 有基线：没有基线无法衡量进步（如「目前自动处理率 20%」）。
2. 可测量：每个 KR 有明确计算方法和数据来源。
3. 有挑战但可达：OKR 达成 60-80% 是合理的。
4. 区分领先与滞后指标：领先（Leading）如自动化率、API 调用量，可实时影响；滞后（Lagging）如客户满意度、收入，反映长期效果。

### 4.4 A/B 测试

```
流量入口 → 随机分流
  ├ 50% 对照组 A（人工处理 / 旧模型）
  └ 50% 实验组 B（AI 处理 / 新模型）
→ 统一指标采集（处理时长、满意度、成本）
→ 统计显著性检验（p-value < 0.05 ?）
   p < 0.05 → 结论有效；否则增加样本量 / 延长实验
```

关键设计要素：

| 要素 | 说明 | 示例 |
|------|------|------|
| 随机化 | 确保两组用户特征分布一致 | 用用户 ID 的 hash 值分流 |
| 样本量 | 基于预期效应大小和统计功效计算 | 检测 10% 差异需每组 2000+ 样本 |
| 实验周期 | 覆盖完整业务周期（至少 1-2 周） | 包含工作日和周末 |
| 主要指标 | 核心指标（不超过 3 个） | 工单处理时长、客户满意度 |
| 护栏指标 | 不能恶化的指标 | 系统可用性、错误率 |

显著性检验：

```python
from scipy import stats
t_stat, p_value = stats.ttest_ind(treatment_scores, control_scores)
if p_value < 0.05:
    print(f"结果统计显著 (p={p_value:.4f})，实验组优于对照组")
```

常见陷阱：

| 陷阱 | 描述 | 如何避免 |
|------|------|---------|
| 辛普森悖论 | 分组看都赢，合起来却输 | 按用户类型分层分析 |
| 选择偏差 | 两组用户特征不一致 | 严格随机分流，分流后检查特征分布 |
| 新奇效应 | 刚用觉得新鲜，之后下降 | 实验周期至少 2 周，观察效果是否衰减 |
| P-hacking | 反复看数据，等到 p<0.05 就停 | 预先确定周期和样本量，不中途看结果 |
| 多指标问题 | 测 20 个指标，碰巧 1 个显著 | 预先确定 1-3 个主要指标 |

### 4.5 Drift Detection

质量退化原因：训练数据老化、用户行为变化、业务规则变更、外部依赖变化。典型案例：上线时分类准确率 95%，3 个月后降至 82%，原因是新增 3 种工单类型而模型没见过。

检测流程：

```
AI 服务输出 → 实时采集每条请求与响应
  → 自动：LLM-as-a-Judge 自动评分 ／ 人工：定期抽样标注
  → 评分 < 阈值 ?
     是 → 触发告警（PagerDuty）→ 评估退化程度
            可接受？ 否 → 自动回滚旧模型 或 全量转人工
                     是 → 标记观察状态，增加抽样频率
     否 → 正常
```

LLM-as-a-Judge 实现（准确性/完整性/有用性各 1-5 分，总体 0-100）：

```python
class QualityScore(BaseModel):
    accuracy: int        # 1-5
    completeness: int    # 1-5
    helpfulness: int     # 1-5
    overall: float       # 0-100

def judge_response(request: str, response: str, ground_truth: str) -> QualityScore:
    prompt = f"作为评估专家，请对以下 AI 回答评分：\n用户请求：{request}\nAI 回答：{response}\n参考答案：{ground_truth}\n请从准确性、完整性、有用性评分（1-5），并给出总体得分（0-100）。"
    return llm.with_structured_output(QualityScore).invoke(prompt)

# 每日自动评估
avg_score = sum(daily_scores) / len(daily_scores)
if avg_score < BASELINE_SCORE * 0.9:   # 低于基线 10% 告警
    alert(f"模型质量下降：当前 {avg_score:.1f}，基线 {BASELINE_SCORE}")
```

告警与自动回滚配置（数值口径）：

```yaml
drift_detection:
  schedule: "*/6h"                # 每 6 小时评估一次
  alert:
    quality_drop_threshold: 10%   # 质量下降超 10% 告警
    volume_threshold: 500         # 评估样本至少 500 条
  auto_rollback:
    conditions:
      - quality_drop > 20% AND duration > 24h
      - error_rate > 15% AND duration > 1h
      - customer_complaint_rate > 5%
  rollback_actions:
    - switch_to_previous_model    # 切回上一稳定版本
    - increase_human_review       # 提升人工审核比例
    - enable_fallback_rules       # 启用规则引擎兜底
```

### 4.6 指标采集架构与看板

```
数据采集：SDK 埋点（延迟、错误率）、业务日志（工单状态、满意度）、成本数据（GPU 小时、API 调用）
  → 数据管道：Kafka 实时数据流 → Flink 实时聚合
  → 存储：TimescaleDB（时序数据）、ClickHouse（分析查询）
  → 展示与告警：Grafana 仪表盘、PagerDuty 告警（告警规则来自 Flink）
```

三张关键仪表盘：

| 仪表盘 | 受众 | 核心指标 | 更新频率 |
|--------|------|---------|---------|
| 技术运营 | 工程师 | 延迟、错误率、GPU 利用率、队列深度 | 实时（10s） |
| 业务运营 | 运营经理 | 自动处理率、人工介入率、处理时长 | 5 分钟 |
| 高管报告 | CEO/CFO | 月度人力节省、ROI、客户满意度变化 | 每日 |

向管理层汇报的四段结构（结论 → ROI → 趋势对比 → 风险与下一步），每段约 30 秒，用数字说话、先说结果再说过程。

### 4.7 指标体系最佳实践

1. 先定义成功再开工：启动时即明确 OKR。
2. 技术 + 业务双仪表盘：工程师看技术，管理层看业务。
3. 基线！基线！基线！：上线前收集至少 2 周基线数据。
4. A/B 测试要严谨：预先确定样本量和周期，不「看看数据再说」。
5. 建立质量基线：上线时做全面质量评估，作为 Drift Detection 基准。
6. 定期质量巡检：每月一次 LLM-as-a-Judge 全量评估。
7. 成本透明化：每次请求成本可计算、可分摊到业务线。
8. 告警分级：P0（自动回滚）、P1（2 小时内响应）、P2（工作日内处理）。

---

## 五、现场可用的速查

### 5.1 关键阈值表

| 场景 | 阈值 | 动作 |
|------|------|------|
| 工单分类置信度路由 | ≥0.9 / 0.7-0.9 / <0.7 | 自动处理 / AI+人工审核 / 直接转人工 |
| 抽检审核抽样比例 | 5%-10% | 随机抽样人工审核 |
| 抽检合格率线 | > 95%（连续 4 周） | 维持自动；低于则切全量审核 |
| AI 服务超时 | 30s | 降级为预设回复 + 创建人工工单 |
| 外部 API 调用超时 | 建议 30s | 超时进入重试 |
| 重试次数上限 | 3 次 | 指数退避 + 抖动；超过入死信队列 |
| API 限流（Kong 示例） | 10000 次/小时 | 按 consumer 限流，Redis 计数 |
| 请求体大小上限 | 1MB | 拒绝超大请求 |
| Webhook 密钥缓存 TTL | 5 分钟（Redis 客户画像） | 降级读缓存 |
| Drift 检测频率 | 每 6 小时 | 自动评分 |
| Drift 告警阈值 | 质量下降 > 10% | 告警 |
| Drift 评估样本量 | ≥ 500 条 | 低于则样本不足 |
| 自动回滚条件 1 | 质量下降 > 20% 且持续 > 24h | 切回旧模型 |
| 自动回滚条件 2 | 错误率 > 15% 且持续 > 1h | 切回旧模型 |
| 自动回滚条件 3 | 客户投诉率 > 5% | 切回旧模型 |
| 自动处理率告警线 | 跌破 50% | 触发告警 |
| A/B 显著性 | p-value < 0.05 | 结论有效；否则加样本 |
| A/B 样本量 | 每组 2000+（检测 10% 差异） | — |
| A/B 实验周期 | 至少 1-2 周 | 覆盖工作日与周末 |
| 基线数据收集 | 至少 2 周 | 上线前完成 |

### 5.2 HITL 模式选择速查

| 判断条件 | 选前置审核 | 选后置审核 | 选抽检审核 |
|---------|-----------|-----------|-----------|
| 自动化率 | 30-50% | 60-80% | 95%+ |
| 执行后果 | 重 | 重 | 轻 |
| AI 质量 | 未知 | 通过率 >90% | 长期稳定 |
| 阶段 | 新系统上线 | 系统成熟期 | 系统稳定期 |

### 5.3 集成关键动作清单

- 所有外部事件 → CloudEvents 统一格式。
- 所有 Webhook → HMAC-SHA256 验签 + `hmac.compare_digest`。
- 所有写操作 → 幂等，业务 ID 作去重键。
- 所有跨系统事务 → 先设计补偿（反向逆序执行）。
- 所有失败 → 重试（429 等 Retry-After；4xx 与业务错误不重试）。
- 所有超限 → 进死信队列，消息带 original_message / error / retry_count / 首次与末次时间 / context。
- 所有外部调用 → 带 X-Request-ID 全链路追踪。
- 一致性选型：支付/库存→强一致；客户信息/状态→最终一致；混合→关键强、非关键最终一致。
- 读同步：CDC（Debezium + Binlog + Kafka）；写并发：乐观锁版本号校验。
- 认证选型：内部调用→API Key；服务间→OAuth2 Client Credentials；高安全→mTLS。

### 5.4 指标映射速查

| 技术指标 | 业务指标 | 财务指标 |
|---------|---------|---------|
| GPU 利用率 | 单次处理成本 | ROI |
| P99 延迟 | 平均处理时长 | 客户流失率 |
| 模型准确率 | 自动处理率、客户满意度 | 月度人力节省 |
| Token 成本 | 单次处理成本 | ROI |
| 幻觉率 | 工单升级率 / 人工介入比例 | — |
| 工具调用成功率 | 自动化流程完成率 | — |

### 5.5 五类角色演进顺序

辅助（门槛/风险最低）→ 增强 → 决策 → 编排 → 替代（门槛/风险最高）。企业落地从辅助起步，逐级推进，每级达标后再进入下一级。

### 5.6 工作流设计检查项

- 每个状态都有出口，无死状态。
- 每个状态有超时转移（如分类 30s）。
- 每步操作幂等。
- 每次状态转移留可审计日志（含时间戳、原因、AI 输入/输出/置信度）。
- 有降级路径（AI 挂了、模型质量下降、CRM 不可用）。
- 有异常归档与事后分析。
- 人工修改的数据回流训练集，形成数据飞轮。
