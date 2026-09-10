# Deep Research — 多智能体深度研究平台

一个基于 Claude API 的多 Agent 研究系统。输入一个问题，自动完成：**拆解 → 搜索 → 深度抓取 → 分析 → 补搜缺口 → 真实验证 → 生成报告**，全程约消耗 10-15 万 tokens。

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
  ▼  并行 asyncio.gather（并发受 Semaphore 限制）
[Searcher ×5]   ── Tavily + Firecrawl + Top-N 深度抓取 ──▶  list[SearchResult]
  │
  ▼  并行 asyncio.gather
[Analyzer ×5]   ── 从搜索结果中提取事实声明 ──▶  list[Claim]
  │
  ▼  若 estimated_rounds>1 且预算充足
[Gap-Fill]      ── 针对 gaps 发起第二轮定向搜索 ──▶  补充 Claims
  │
  ▼  串行
[Verifier]      ── 交叉验证 + 低置信度声明定向搜索 ──▶  VerificationOutput
  │
  ▼  串行
[Writer]        ── 合成带引用的完整报告 ──▶  FinalReport
```

每个 Agent 独立调用 Claude API，拥有自己的上下文窗口，通过 Pydantic 模型定义结构化 I/O 契约。

### 关键能力

| 能力 | 说明 |
|------|------|
| **深度抓取** | 对 Top-N 高分结果调用 Firecrawl scrape，获取全文 |
| **真实验证** | Verifier 对低置信度/单源声明发起独立搜索再判定 |
| **第二轮补搜** | Analyzer 发现的 gaps 触发定向补充搜索 |
| **断点续跑** | 每阶段产物落盘到 `.runs/<run_id>/`，可事后检查 |
| **自动重试** | LLM 和搜索 API 均带指数退避重试 |
| **并发限制** | LLM 并发默认 3，搜索并发默认 2，避免限流 |
| **预算降级** | Token 耗尽时自动生成最小报告，不白跑 |

## 快速开始

### 1. 安装依赖

```bash
# 推荐：以包方式安装
pip install -e ".[dev]"

# 或者只装运行依赖
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

启动时会校验 key 是否存在，缺失则立即报错退出。

### 3. 运行

```bash
# 直接指定问题
python main.py "Zephyr RTOS vs FreeRTOS 在 IoT 场景的对比"

# 保存报告到文件
python main.py "2025年边缘计算最新趋势" --output report.md

# 自定义 token 预算 + 详细日志
python main.py "神经网络压缩技术综述" --budget 300000 --verbose

# 指定 run-id（便于关联 checkpoint）
python main.py "量子计算进展" --run-id my-run-001 --output report.md
```

### 4. 运行测试

```bash
python -m pytest tests/ -v
```

## 核心技术点

| 技术 | 说明 |
|------|------|
| **Tool-use 结构化输出** | Pydantic schema 作为 `tool_choice` 的 `input_schema`，强制 Claude 返回精确 JSON |
| **JSON 字符串自动修复** | 嵌套字段序列化为字符串时自动 `json.loads`；字面量 `\n` 还原为真实换行 |
| **async 并行 Agent** | Searcher 和 Analyzer 通过 `asyncio.gather` 并行执行 |
| **全局 Token 预算** | 所有 Agent 共享 `TokenBudget`，超限自动降级 |
| **Fallback 降级** | 任一 Agent 失败时自动降级，不中断整个流程 |
| **指数退避重试** | LLM / 搜索 API 对 429/5xx 自动重试 |
| **Semaphore 并发控制** | 防止并行调用触发限流 |

## 项目结构

```
deep-research/
├── main.py                         # CLI 薄包装
├── pyproject.toml                  # 包配置 + entry point
├── .env.example                    # API key 模板
├── .gitignore
├── tests/                          # 单元测试
│   ├── conftest.py
│   ├── test_coercion.py            # JSON 修复 / 换行还原
│   └── test_budget_config.py       # Token 预算 / 配置
└── src/deep_research/
    ├── cli.py                      # CLI 入口（argparse + logging）
    ├── config.py                   # pydantic-settings 配置
    ├── models.py                   # 所有 Pydantic 数据模型
    ├── llm.py                      # Anthropic SDK 封装 + 重试 + Token 预算
    ├── search.py                   # Tavily + Firecrawl + 深度抓取
    ├── pipeline.py                 # 管道编排 + checkpoint + Rich UI
    └── agents/
        ├── base.py                 # BaseAgent + JSON 修复 + 换行还原
        ├── orchestrator.py         # 问题拆解
        ├── searcher.py             # 并行搜索 + Top-N 深度抓取
        ├── analyzer.py             # 事实提取
        ├── verifier.py             # 交叉验证 + 定向搜索
        └── writer.py               # 报告生成
```

## 配置项

可通过环境变量（`DR_` 前缀）覆盖：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DR_LLM_CONCURRENCY` | 3 | LLM 并行调用上限 |
| `DR_SEARCH_CONCURRENCY` | 2 | 搜索并行调用上限 |
| `DR_LLM_MAX_RETRIES` | 3 | LLM 调用最大重试次数 |
| `DR_SEARCH_MAX_RETRIES` | 2 | 搜索 API 最大重试次数 |
| `DR_DEEP_SCRAPE_TOP_N` | 3 | 深度抓取的 Top-N 结果数 |
| `DR_VERBOSE` | false | 启用 DEBUG 日志 |
| `DR_CHECKPOINT_DIR` | .runs | 中间产物目录 |

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
