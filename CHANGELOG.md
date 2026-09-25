# Changelog

本文件的格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.0.0] - 2026-09-25

首个公开发布版本。

### 包含

- `SKILL.md`：身份定位、六条铁律、九步工作流与文件路由、现场速查阈值、交付红线清单、完成度自查
- `references/`：13 份现场手册
  - 客户进场与交付：`01-client-engagement`
  - AI 应用工程：`02-agentic-rag`
  - 模型与硬件：`03-model-gpu`
  - 推理优化：`04-inference`
  - 分布式与成本：`05-distributed-production-cost`
  - 排障与工具：`06-playbooks-resources`
  - 本体建模与现场打法：`07-palantir-fde-playbook`
  - 真实部署案例：`09-deployment-cases`
  - 方法论与中国实践：`10-fde-methodology`
  - 业务流程与指标：`11-business-workflows`
  - 合规与安全：`12-compliance-security`
  - 采纳运营与团队：`13-adoption-and-team`
  - 基础认知与术语表：`14-fde-basics-and-glossary`
- `assets/templates.md`：8 个交付物模板骨架 + 第 9 节一组填好的样例
- 工程件：`LICENSE`(MIT)、`README.md`、`THIRD-PARTY-NOTICE.md`

### 说明

- 部分内容改写自开源项目 [fde-learning](https://github.com/luoboask/fde-learning)（MIT），逐项声明见 `THIRD-PARTY-NOTICE.md`。
- 本版本**不含**具体的客户项目案例，也**不含**付费资料内容。
- 未覆盖范围（厂商报价、行业专属案例、模型训练与微调、等保/备案申报流程）已在 README 中列明。

### 已知限制

- 手册中的数值与阈值多来自特定硬件与场景的实测或公开调研，**仅作数量级参考**，落地前须自行复测。
- 合规章节**不构成法律意见**。
