# Deep Research — 多智能体深度研究平台

一个基于 Claude API 的多 Agent 研究系统。输入一个问题，自动完成：**拆解 → 搜索 → 分析 → 验证 → 生成报告**，全程约消耗 10-15 万 tokens。

## 它能做什么

```
你输入: "RISC-V 嵌入式 AI 加速器 2025 年最新进展"

它输出:
├── Executive Summary（带引用的研究摘要）
├── 5 个分主题章节（芯片产品 / 指令集扩展 / 市场 / 应用 / 学术）
├── 30 条引用来源
└── 方法论说明
```

## 架构

```
用户问题
  │
  ▼
[Orchestrator]  ── Sonnet ──▶  ResearchPlan（拆成 3-5 个子问题）
  │
  ▼  并行 asyncio.gather
[Searcher ×5]   ── 同时搜索 Tavily + Firecrawl ──▶  list[SearchResult]
  │
  ▼  并行 asyncio.gather
[Analyzer ×5]   ── 从搜索结果中提取事实声明 ──▶  list[Claim]
  │
  ▼  串行
[Verifier]      ── 交叉验证所有声明 ──▶  VerificationOutput
  │
  ▼  串行
[Writer]        ── 合成带引用的完整报告 ──▶  FinalReport
```

每个 Agent 独立调用 Claude API，拥有自己的上下文窗口，通过 Pydantic 模型定义结构化 I/O 契约。

## 快速开始

### 1. 安装依赖

```bash
pip install anthropic pydantic-settings httpx rich
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 key：

```bash
cp .env.example .env
```

```env
DR_ANTHROPIC_API_KEY=sk-ant-xxx
DR_ANTHROPIC_BASE_URL=https://api.anthropic.com
DR_TAVILY_API_KEY=tvly-xxx
DR_FIRECRAWL_API_KEY=fc-xxx
```

| Key | 来源 | 用途 |
|-----|------|------|
| `DR_ANTHROPIC_API_KEY` | [Anthropic Console](https://console.anthropic.com/) | 调用 Claude 模型 |
| `DR_ANTHROPIC_BASE_URL` | 默认 `https://api.anthropic.com` | API 地址（可改为代理） |
| `DR_TAVILY_API_KEY` | [Tavily](https://tavily.com/) | 网页搜索（免费额度） |
| `DR_FIRECRAWL_API_KEY` | [Firecrawl](https://firecrawl.dev/) | 深度网页抓取 |

### 3. 运行

```bash
# 直接指定问题
python main.py "Zephyr RTOS vs FreeRTOS 在 IoT 场景的对比"

# 保存报告到文件
python main.py "2025年边缘计算最新趋势" --output report.md

# 自定义 token 预算
python main.py "神经网络压缩技术综述" --budget 300000 --output report.md
```

## 核心技术点

| 技术 | 说明 |
|------|------|
| **Tool-use 结构化输出** | Pydantic schema 作为 `tool_choice` 的 `input_schema`，强制 Claude 返回精确 JSON |
| **JSON 字符串自动修复** | 代理返回的嵌套字段序列化为字符串时，自动 `json.loads` 修复 |
| **async 并行 Agent** | Searcher 和 Analyzer 通过 `asyncio.gather` 并行执行 |
| **全局 Token 预算** | 所有 Agent 共享 `TokenBudget`，超限自动停止 |
| **Fallback 降级** | 任一 Agent 失败时自动降级，不中断整个流程 |

## 项目结构

```
deep-research/
├── main.py                         # CLI 入口
├── .env.example                    # API key 模板
├── .gitignore
└── src/deep_research/
    ├── config.py                   # pydantic-settings 配置
    ├── models.py                   # 所有 Pydantic 数据模型
    ├── llm.py                      # Anthropic SDK 封装 + Token 预算
    ├── search.py                   # Tavily + Firecrawl HTTP 搜索
    ├── pipeline.py                 # 管道编排 + Rich 终端 UI
    └── agents/
        ├── base.py                 # BaseAgent 抽象类
        ├── orchestrator.py         # 问题拆解
        ├── searcher.py             # 并行搜索
        ├── analyzer.py             # 事实提取
        ├── verifier.py             # 交叉验证
        └── writer.py               # 报告生成
```

## 自定义

修改 `config.py` 中的模型配置：

```python
model_haiku: str = "claude-sonnet-4-6"    # 轻量任务（搜索筛选）
model_sonnet: str = "claude-sonnet-4-6"   # 主力任务（分析、验证）
model_opus: str = "claude-sonnet-4-6"     # 高质量任务（报告生成）
```

如有直接 Anthropic API 权限，可改为 `claude-haiku-4-5` / `claude-opus-4-7` 以区分成本。

## License

MIT
