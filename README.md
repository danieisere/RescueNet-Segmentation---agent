# RescueNet 应急响应智能体（RERA）

RescueNet Emergency Response Agent（RERA）是一个面向灾后无人机图像的多模态应急响应智能体。大语言模型识别本轮目标，程序根据真实状态和阶段依赖执行分割、统计、优先级、综合分析和报告。

项目当前支持：

- 多模型集成语义分割：UNet、PSPNet、DeepLabV3 选择当前输入的最佳结果。
- 原始类别掩膜统计：计算类别像素数量和占整图比例。
- 规则优先级评分：根据建筑损坏和道路阻塞结果给出 CRITICAL/HIGH/MEDIUM/LOW 等级。
- 多模态综合分析：同时读取原图、彩色分割图和半透明叠加图，并结合统计及优先级结果生成解释或救援建议。
- HTML/Markdown 报告：报告使用当前综合分析正文，独立保存图片、掩膜、统计、优先级和分析数据。
- 多轮会话：原始消息、有效事实、历史摘要、检查点和每轮产物按会话保存，可退出后恢复。
- 上下文自动压缩：超过应用预算后压缩较早讨论，保留原始消息、当前请求、有效事实和方案来源。

## 快速运行

先安装依赖、准备配置和模型权重：

```bash
pip install -r requirements.txt
copy config.yaml.example config.yaml
```

编辑 `config.yaml` 或使用环境变量提供兼容 OpenAI 接口的大模型凭据。模型必须支持图片输入；三个分割权重应位于 `runs/`，文件名与 `ensemble.models` 配置一致。

### 默认多轮模式

直接运行即可新建一个会话：

```bash
python agent_cli.py
```

也可以在启动时提供首张图片和首轮问题：

```bash
python agent_cli.py \
  --image-folder ./RescueNet/test/test-org-img/10794.jpg \
  --question "先统计各类别像素占比"
```

命令会进入交互循环。每次普通输入都是同一会话的新一轮；输入 `/exit` 或 `退出` 会保存并退出，之后可以使用终端显示的 `session_id` 恢复：

```bash
python agent_cli.py --session-id 20261031_165539
```

会话命令：

| 命令 | 作用 |
|---|---|
| `/exit`、`退出` | 保存并退出当前会话 |
| `/new` | 创建新会话并保留旧会话 |
| `/image 图片路径` | 切换当前图片 |
| `/sessions` | 列出会话 |
| `/help` | 查看命令 |

`--chat` 仍可使用，但它是兼容参数；多轮现在是默认模式。会话 ID 使用 UTC+8 的 `YYYYMMDD_HHMMSS` 格式，默认保存到根目录 `message/`。

### 单轮或批量模式

需要执行一次后退出，或者需要处理一个图片目录时，必须显式添加 `--single-turn`：

```bash
# 默认生成救援建议，不自动生成报告
python agent_cli.py --single-turn \
  --image-folder ./RescueNet/test/test-org-img/10801.jpg

# 只统计类别像素和占比
python agent_cli.py --single-turn \
  --image-folder ./RescueNet/test/test-org-img/10794.jpg \
  --question "只统计各类别像素和占比，不要报告"

# 结合三张图做视觉解释
python agent_cli.py --single-turn \
  --image-folder ./RescueNet/test/test-org-img/10836.jpg \
  --question "结合原图、分割图和叠加图解释道路损坏分布，不要救援建议，不要报告"

# 生成 HTML 报告
python agent_cli.py --single-turn \
  --image-folder ./RescueNet/test/test-org-img/10808.jpg \
  --output assessment.html

# 批量处理目录；每张图独立执行、独立归档
python agent_cli.py --single-turn \
  --image-folder ./RescueNet/test/test-org-img
```

单轮默认使用 LangGraph。保留的 ReAct 引擎需要显式指定：

```bash
python agent_cli.py --single-turn --engine react \
  --image-folder ./RescueNet/test/test-org-img/10838.jpg \
  --output assessment.html
```

`--output` 仅用于单轮/批量报告文件名。会话模式应在对话中提出“生成 HTML 报告”或“生成 Markdown 报告”。

## 当前执行架构

多轮模式由两层图组成：

```text
外层会话图
  接收消息 → 准备上下文/压缩 → 识别意图 → 更新事实 →
  普通回答或调用内层分析图 → 保存消息、状态和产物 → 等待下一轮

内层分析图
  校验 → 目标识别 → 分割 → 统计 → 优先级 → 多模态分析 → 报告 → 归档
```

LLM 只负责输出经过协议校验的目标；阶段路由由程序根据目标、当前状态、依赖版本和结果完整性决定。统计请求不会自动调用报告节点，报告请求会自动补齐所需的上游阶段。完整图、条件边和状态约束见：

- [LANGGRAPH_ARCHITECTURE.md](LANGGRAPH_ARCHITECTURE.md)
- [LANGGRAPH_WORKFLOW.md](LANGGRAPH_WORKFLOW.md)
- [MULTITURN_ARCHITECTURE.md](MULTITURN_ARCHITECTURE.md)

ReAct 控制器仍保留 Thought → Action → Observation 循环，但它只在 `--single-turn --engine react` 下使用；它不代表默认 LangGraph 的节点事件。

## 多模态与分割可视化

分割阶段会保存三种图片：原图、彩色分割图、原图与分割结果的半透明叠加图。三张图片使用同一套类别颜色和图例；掩膜映射回原图时记录尺寸关系，类别掩膜缩放使用最近邻插值，避免生成不存在的类别编号。统计和优先级始终读取原始类别数组，不从可视化图片反推数值。

视觉解释和救援建议由 `MultimodalAnalysisService` 完成：

- `interpretation`：结合三图、图例和统计，输出视觉观察、空间关系、统计解释、疑点和局限。
- `rescue`：在上述证据基础上增加规则优先级和有依据的行动建议。

图片发送给模型时默认限制长边为 1536 并保持比例；报告归档的图片保持可复核的全尺寸版本。模型输出不被当作现场核实事实，像素比例不等于建筑数量、受灾面积、被困人数或道路实际通行率。

## 产物与会话目录

单轮/批量运行的产物根目录是：

```text
output/artifacts/<run_id>/
  request.json
  events.jsonl
  manifest.json
  result.json
  items/<item_id>/
    request.json
    events.jsonl
    manifest.json
    result.json
    images/
      original.jpg
      segmentation.png
      overlay.png
    data/
      mask.npy
      visualization.json
      statistics.json
      priority.json
      analysis.json
    report/
      <report_name>.html 或 <report_name>.md
```

多轮会话按会话 ID 保存到：

```text
message/<session_id>/
  session.json
  memory.sqlite
  checkpoints.sqlite
  messages.jsonl
  facts.json
  summaries.json
  inputs/<image_id>/
  turns/<turn_id>/
    request.json
    result.json
    events.jsonl
    artifacts/<run_id>/items/<item_id>/...
```

`memory.sqlite` 是业务记忆的权威来源；JSON/JSONL 是可读导出。原始消息永久保留，事实更正通过 `active` 与 `superseded` 记录，历史摘要只用于压缩较早讨论。历史轮次的报告和图片不会被后续轮次覆盖。

## 上下文压缩与恢复

默认应用压缩阈值为 8000，压缩目标为 6000。最近轮次是优先保留范围，但预算不足时会动态扩大压缩范围；当前请求、有效事实、相关方案和来源消息必须保留。长消息会分段整理，压缩失败不会删除原始消息或覆盖有效摘要。

终端会显示 `compression_start`、`compression_retry` 和 `compression_end` 等事件。这里的 8000 是应用层预算，不等于服务商的真实上下文窗口；无法使用本地 tokenizer 时，系统会明确标注保守估算。外层检查点保存会话状态，分割掩膜、Base64 图片和模型对象不写入检查点。

## 配置

主要配置节：

| 配置 | 作用 |
|---|---|
| `llm` | OpenAI 兼容服务、模型和凭据 |
| `ensemble` | 分割模型列表、权重目录和置信度评分参数 |
| `priority_weights` | 救援优先级计算权重 |
| `multimodal` | 图片传输尺寸、模型输出预算、超时和 JSON 重试 |
| `artifacts` | 单轮/批量产物根目录 |
| `conversation` | 会话目录、压缩预算、摘要和回答预算 |
| `graph` | LangGraph 目标识别重试、超时和最大步骤数 |
| `agent.engine` | 默认引擎，当前为 `langgraph` |

`config.yaml` 和 `.env` 不应提交真实 API Key。也可以设置 `RERA_CONFIG` 指向项目外的配置文件。

## 验证

项目原有回归测试和多轮验收脚本均可在项目外运行目录中执行。常用检查命令：

```bash
python -m pytest -q
python -m compileall -q modules
```

Windows 沙箱或受限环境下，如果 pytest 使用系统临时目录失败，应先确认目录权限；这类环境错误不能当作业务逻辑通过。

## 数据集和训练

RescueNet 数据集不随仓库提供。按 `verify_dataset.py` 的提示准备数据后，可以运行：

```bash
python verify_dataset.py
python train.py
python test.py
```

训练、评估和数据集脚本与 Agent 调度层独立；本项目当前改造集中在多模态分析、LangGraph 编排、多轮记忆和产物归档。