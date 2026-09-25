# 第三方内容与许可声明（THIRD-PARTY NOTICE）

本仓库的绝大部分内容，是在公开信息与开源资料的基础上**重新组织与表述**的产物。
本文件说明来源、授权与处理方式，同时作为合规凭证。

---

## 一、明确引用的开源项目

### fde-learning

- **仓库**：https://github.com/luoboask/fde-learning
- **许可**：MIT（作者在仓库 README 中声明；仓库根目录未附 `LICENSE` 文件）
- **使用方式**：本仓库的下列手册，其**内容框架、阈值、对比表与决策规则**大量改写自该项目的 `docs/` 目录，并按其 MIT 条款使用与再分发。

| 本仓库文件 | 主要来源（该项目的 docs 模块） |
|---|---|
| `references/02-agentic-rag.md` | `06-ai-engineering` 及公开技术共识 |
| `references/03-model-gpu.md` | `02-model-architecture`、`03-gpu-basics` |
| `references/04-inference.md` | `04-inference-optimization` |
| `references/05-distributed-production-cost.md` | `05-distributed-inference`、`07-production-deployment`、`08-cost-operations` |
| `references/06-playbooks-resources.md` | `09-labs`、`15-resources` |
| `references/09-deployment-cases.md` | `12-interview`、`15-resources` |
| `references/11-business-workflows.md` | `10-business-workflows` |
| `references/12-compliance-security.md` | `11-compliance-security` |
| `references/13-adoption-and-team.md` | `13-change-adoption`、`14-team-building` |
| `references/14-fde-basics-and-glossary.md` | `01-basics` |
| `assets/templates.md` | 自建骨架 |

**MIT 条款要求保留版权与许可声明**——本文件即履行该义务。该项目的原始版权归其作者所有。

---

## 二、独立撰写的内容

`references/01-client-engagement.md`、`references/07-palantir-fde-playbook.md`、`references/10-fde-methodology.md`、`SKILL.md`，以及 `assets/templates.md` 中的交付物骨架：

- 其**表述与结构为独立撰写**；
- 其中涉及的事实性数据（模型参数与显存、案例前后指标、公开市场薪酬调研、公开产品信息）与公开概念（Ontology 的对象 / 属性 / 链接 / 动作四件套、Palantir 的公开产品与公开术语），**均属不受版权保护的公开信息**，可直接使用。

---

## 三、未收录的内容（有意排除）

为避免版权与合规风险，本仓库**有意不收录**：

1. 任何第三方付费课程、付费文档、付费社群的原文或其衍生内容；
2. 任何具体客户项目的可识别信息（客户名称、对象命名、字段命名、金额、业务标识）；
3. 任何需要许可证才能获取的商业数据集。

---

## 四、使用时请注意

- 本仓库中的**数值与阈值**多为特定硬件/场景下的实测或调研结果，**仅作数量级参考**。落地前请用你自己的数据复测。凡是标注 `[需确认]` 的地方，务必现场验证。
- 涉及**合规与法律**的内容（数据出境、个人信息保护、行业监管）**不构成法律意见**，请以你所在地的现行法规与专业顾问意见为准。

---

## 五、引用格式

如果你在作品中引用本仓库：

```
FDE：前沿部署工程师工作台（AI 落地手册）
部分内容改写自 fde-learning (https://github.com/luoboask/fde-learning, MIT)
```

---

*本文件为依据 MIT 许可使用第三方开源内容所必需的声明文件。若你发现本仓库中有内容侵犯了你的权益，请提 Issue，我们会核实后处理。*
