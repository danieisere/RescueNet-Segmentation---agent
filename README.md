# RescueNet 应急响应智能体 (RERA)

CLI 默认进入多轮对话，直接运行 `python agent_cli.py` 即可；带单张图片和 `--question` 时作为首轮输入，之后继续等待提问。`--chat` 是可省略的兼容参数。单轮或批量运行需显式添加 `--single-turn`，ReAct 示例为 `--single-turn --engine react`。

默认使用 **LangGraph**：LLM 识别本轮单图目标，程序根据真实状态及依赖控制分割、统计、优先级、三图综合分析和可选报告。`--engine react` 保留原 ReAct 完整处理流程。根目录 [LANGGRAPH_ARCHITECTURE.md](LANGGRAPH_ARCHITECTURE.md) 保存架构图、状态及验收约定。

```bash
# 单轮：给出救援建议，保存独立产物，不自动生成报告
python agent_cli.py --single-turn --image-folder ./RescueNet/test/test-org-img/10801.jpg

# 仅统计：不会调用优先级、综合分析或报告节点
python agent_cli.py --single-turn --image-folder ./RescueNet/test/test-org-img/10794.jpg --question "只统计各类别像素和占比，不要报告"

# 三图解释：不强制计算优先级
python agent_cli.py --single-turn --image-folder ./RescueNet/test/test-org-img/10836.jpg --question "结合三图解释道路损坏分布，不要救援建议，不要报告"

# 显式生成报告；也支持 --output assessment.md
python agent_cli.py --single-turn --image-folder ./RescueNet/test/test-org-img/10808.jpg --output assessment.html

# 保留的 ReAct 引擎
python agent_cli.py --single-turn --engine react --image-folder ./RescueNet/test/test-org-img/10838.jpg --output assessment.html
```

安装使用 `pip install -r requirements.txt`；需要支持图片输入的 OpenAI 兼容模型服务及本地三个分割权重。CLI 通过 LangGraph 原生 custom/updates 流显示节点开始、结束、重试、归档及安全状态摘要。公开的目标判断说明不是模型内部完整思维链。

默认产物位于 `output/artifacts/<run_id>/items/<item_id>/`：`images/` 包含三图，`data/` 包含原始 `mask.npy`、图例/映射及实际完成阶段的 JSON，`report/` 只在需要时创建。run/item 均有请求、结果、manifest 和事件记录；掩膜统计与图像展示独立。`artifacts.output_dir` 可修改根目录；旧自定义 `report.output_dir` 在没有新配置时仍生效。旧 `output/reports/` 已按哈希校验迁入 `output/artifacts/legacy_reports/`。

输入目录按图片分别执行并复用分割模型，各图状态独立，不支持跨图排名。单轮退出码：全部完成 0，有执行失败 1，仅含澄清/不支持而无失败 2。失败保留已完成的阶段产物，不将 END 或部分产物当作整体成功。

多轮模式采用外层会话 LangGraph + 内层分析 LangGraph。根目录 [MULTITURN_ARCHITECTURE.md](MULTITURN_ARCHITECTURE.md) 保存会话架构图、状态、目录及恢复约定。

```bash
# 新建会话，可先提供图片和首轮问题；之后在终端连续提问
python agent_cli.py --image-folder ./RescueNet/test/test-org-img/10794.jpg --question "先统计各类别像素占比"

# 恢复会话，继续原来的轮次编号；ID 替换为程序显示的真实值
python agent_cli.py --session-id 20261031_153042

# 列出已有会话；纯文本对话直接运行 python agent_cli.py
python agent_cli.py --list-sessions
```

交互命令：`/exit` 或 `退出` 暂停并保留会话，`/new` 新建会话，`/image 图片路径` 切图，`/sessions` 列表，`/help` 帮助。每次普通输入是同一会话的一轮；关闭后可恢复。多轮默认目录为 `message/<YYYYMMDD_HHMMSS>/`（UTC+8 秒级 ID），同秒新建会等待下一秒避免重名。`--message-root` 可覆盖会话根目录。会话模式使用 LangGraph；原有单轮 ReAct 保留。

每个会话的 `memory.sqlite` 保存原始消息、active/superseded 事实、摘要和产物索引；JSON/JSONL 为可重建的可读导出。`inputs/` 保存原图，`turns/<轮次>/artifacts/` 保存该轮真正生成的三图、真实掩膜、统计、分析和报告。历史轮次不改写；普通追问按需复用经过哈希及依赖签名验证的成果。现场补充改变建议，不修改像素统计。`output/` 继续保存独立单轮/批量及已有训练输出，不自动迁移历史成果。

超过 8000 文本 token 后自动压缩，默认目标 6000；最近 4 轮是优先保留范围，必要时自动压缩较早的近期轮次，保留本轮请求、有效事实、方案内容及来源。原始消息不删除，终端显示压缩开始与完成事件，随后继续本轮任务。“把该方案生成报告”会携带相关方案及原消息引用，助手建议与现场事实分别保存。长消息分段压缩，失败不覆盖有效摘要。使用本地缓存可用的 tiktoken 编码，未知模型编码会明确采用假设或保守 UTF-8 估算；缺少缓存时不隐式下载阻塞终端。8000 是应用压缩阈值，实际模型窗口可通过 `conversation.model_context_window` 配置；图片预算单独控制。外层 SQLite Checkpointer 只保存会话状态；内层数组不进入检查点。恢复支持继续会话及验证复用已落盘成果，不承诺内层计算节点精确断点续跑。当前不包含网页或多人同时写同一会话。

## 1. 项目概述

### 1.1 项目名称
**RescueNet Emergency Response Agent (RERA)**

### 1.2 项目定位
基于 LangGraph 默认调度与兼容 ReAct 调度的多模态灾害应急响应智能体，复用分割模型、规则计算与可配置大语言模型，实现从图像感知到证据解释和救援建议的处理流程。

### 1.3 核心价值
- **解决痛点**: 灾后人工评估效率低，黄金救援时间宝贵
- **技术创新**: LLM控制器 + Ensemble视觉感知 + 规则引擎决策
- **实用价值**: 按需生成结构化救援报告和独立档案；尚无可复现基准支持具体效率提升百分比。

---

## 2. 系统架构

### 2.1 整体架构

当前默认图以 [最终架构图](LANGGRAPH_ARCHITECTURE.md#3-核心架构图) 为准。下面保留的 ReAct 控制器示意用于解释兼容引擎。

```
┌─────────────────────────────────────────────────────┐
│              用户交互层 (CLI/Web)                     │
│  输入: 灾区图像文件夹 + 自然语言问题                  │
│  输出: 应急救援报告 + 可视化结果                      │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         LLM控制器层 (DeepSeek API)                    │
│  ┌─────────────────────────────────────────────┐   │
│  │  ReAct循环引擎                               │   │
│  │  - Thought: 推理下一步行动                    │   │
│  │  - Action: 选择合适的Tool                   │   │
│  │  - Observation: 处理工具返回结果             │   │
│  └─────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────┘
                   │ 调用
┌──────────────────▼──────────────────────────────────┐
│           工具层 (Tools)                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │Ensemble  │ │Statistics│ │Priority  │            │
│  │Segment   │ │Analyzer  │ │Calculator│            │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘            │
│       │            │            │                   │
│  ┌────▼────────────▼────────────▼─────┐             │
│  │   基础分割模型层 (PyTorch)          │             │
│  │   UNet │ PSPNet │ DeepLabV3        │             │
│  └────────────────────────────────────┘             │
└─────────────────────────────────────────────────────┘
```

### 2.2 核心组件

| 组件 | 职责 | 技术实现 |
|------|------|---------|
| **LLM 目标识别 / LangGraph** | 校验目标、计算依赖、根据真实状态流式执行 | 公共 LLMClient + StateGraph |
| **Ensemble Segment Tool** | 多模型融合分割，选择最优结果 | PyTorch + 置信度评分 |
| **Statistics Analyzer** | 统计各类别数量、面积占比 | NumPy/Pandas |
| **Priority Calculator** | 计算救援优先级评分 | 加权算法 |
| **Multimodal Analysis** | 结合原图、分割图、叠加图和工具数据生成综合结论 | 独立分析服务 + OpenAI 兼容多模态接口 |
| **Report Generator** | 在本次 item 中生成 HTML/Markdown 报告 | 复用统一综合结论与共享三图 |

---

## 3. 技术栈

### 3.1 核心技术

```yaml
深度学习框架:
  - PyTorch >= 1.8.0
  - Torchvision >= 0.9.0
  
大语言模型:
  - DeepSeek API (deepseek-chat)
  - OpenAI SDK兼容接口
  
图像处理:
  - OpenCV-Python >= 4.5.0
  - Albumentations >= 1.0.0
  - Pillow >= 8.0.0
  
科学计算:
  - NumPy >= 1.19.0
  - Pandas >= 1.3.0
  
报告生成:
  - Jinja2 >= 3.0.0 (模板引擎)
  - WeasyPrint >= 53.0 (PDF生成)
  
辅助工具:
  - tqdm >= 4.60.0
  - tabulate >= 0.8.0
```

---

## 4. 核心算法设计

### 4.1 Ensemble融合策略（评分最高者胜出）

**设计理念**: 
- RescueNet数据集较小（仅100张训练图像），单个模型泛化能力有限
- 通过Ensemble融合三个模型，利用多样性提升鲁棒性

**评分公式**:
```
Score = mean(max_prob) - λ × normalized_entropy

其中:
- mean(max_prob): 预测类别的最大概率的平均值（确定性）
- normalized_entropy: 归一化的预测熵（一致性）
- λ: 熵惩罚权重（默认0.3）
```

**优势**:
- ✅ 自动选择最可靠的模型结果
- ✅ 无需人工干预
- ✅ 可解释性强（评分透明）

### 4.2 优先级计算算法

**优先级公式**:
```
Priority_Score = w1 × N_destroyed_pixels + w2 × N_major_pixels + w3 × N_minor_pixels + w4 × N_road_blocked_pixels

权重配置:
- w1 (完全毁坏): 0.6
- w2 (严重损坏): 0.3
- w3 (轻微损坏): 0.1
- w4 (道路阻塞): 0.2
```

**分级标准**:
- 🔴 CRITICAL: Score > 10000（立即救援）
- 🟠 HIGH: Score > 5000（优先处理）
- 🟡 MEDIUM: Score > 2000（常规处理）
- 🟢 LOW: Score ≤ 2000（持续关注）

### 4.3 ReAct范式实现

**工作流程**:
```
1. LLM接收用户查询
2. Thought: 分析当前情况，决定下一步
3. Action: 选择工具并给出参数（JSON格式）
4. Observation: 执行工具，获取结果
5. 重复2-4步，直到LLM输出Final Answer
```

**示例流程**:
```
Thought: 我需要先对图像进行分割处理
Action: ensemble_segment
Action Input: {"image_path": "/path/to/image.jpg"}

Observation: {"best_model": "unet", "scores": {...}}

Thought: 现在需要统计损坏情况
Action: analyze_statistics
Action Input: {"seg_results": [...]}

...

Final Answer: 根据分析，发现以下紧急情况：
1. 当前影像中预测为完全毁坏建筑的像素占比为...
2. 建议优先派遣救援队到...
```

---


### 4.4 当前多模态综合分析流程

当前 LangGraph CLI 按目标选择阶段；完整报告和兼容 ReAct 依次执行：分割 → 统计 → 优先级 → 多模态综合分析 → 报告。仅请求救援建议则到综合分析为止。综合分析完成后，可选报告与最终回答复用同一份结论。

- 分割结果中的 `visualizations` 提供原图、彩色分割图、半透明叠加图、完整类别图例和尺寸映射。三图映射到原图尺寸；统计仍读取原始类别掩膜。
- `modules/agent/llm_client.py` 统一处理 Responses 和 Chat Completions 的文字/图片消息。调度只读文字，综合分析通过独立上下文接收真实的三个图片内容块。
- `MultimodalAnalysisService.analyze()` 已被 LangGraph 节点和 ReAct 共同复用；`interpretation` 使用三图与统计，`rescue` 额外要求真实优先级及有依据的救援建议。
- 分析输出区分视觉观察、统计解释、疑点、行动建议及依据和局限；定量统计表由程序根据工具结果生成。视觉观察不等于现场确认，分割像素不等于建筑栋数。
- 报告目录会同时保存 HTML/Markdown 报告和同名 `.analysis.json`。JSON 保存结构化结论、真实数据来源、图片尺寸/字节数/摘要哈希和调用耗时，不保存 Base64 图片正文或凭据。
- 上游重新计算会使下游分析失效；同一上游状态重复调用分析会复用已有结果。图片、分析接口或结构化输出失败会保留已完成状态并明确报错，CLI 以非零退出码结束失败任务。

配置中的 `llm.model` 和兼容服务必须实际支持图片输入；旧示例中的纯文本模型配置需要替换。可选 `multimodal` 配置默认使用独立的 4096 输出 token、180 秒单次请求超时、`auto` 图片细节级别，JSON 结构错误最多尝试两次。传输图片默认限制长边为 1536、保持宽高比，分割图仍使用最近邻；报告保存全尺寸三图，分析 JSON 单独记录来源尺寸、传输尺寸和缩放比例。旧配置文件无需增加此配置节。此阶段尚未迁移 LangGraph 调度。

## 5. 使用指南

### 5.1 安装依赖

```bash
pip install -r requirements.txt
```

### 5.2 配置DeepSeek API Key

复制配置文件模板并填入你的 API Key：

```bash
cp config.yaml.example config.yaml
```

编辑 `config.yaml`，将 `api_key` 替换为你的真实 Key：

```yaml
llm:
  api_key: "sk-your-deepseek-api-key-here"  # ← 替换为你的Key
```


**获取API Key**: https://platform.deepseek.com/

### 5.3 模型训练与评估流程

#### Step 0: 模型训练（前置条件）⚠️

> ⚠️ **数据集需自行下载**：RescueNet 不包含在本仓库中，请先前往 [BinaLab/RescueNet](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation) 下载，按以下结构放置：
> ```
> RescueNet/
> ├── train/train-org-img/   # 训练原始图像
> ├── train/train-label-img/ # 训练标签掩码
> ├── test/test-org-img/     # 测试原始图像
> └── test/test-label-img/   # 测试标签掩码
> ```
> 放置后运行 `python verify_dataset.py` 验证结构。数据集共 11 类、4,494 张高分辨率 UAV 航拍图像。

#### Step 1: 训练模型（必须先完成）

```bash
# 验证数据集
python verify_dataset.py

# 训练三个模型（UNet, DeepLabV3, PSPNet）
python train.py
```

训练完成后，权重文件保存在 `./runs/` 目录。

#### Step 2: 批量评估模型性能

使用 `test.py` 对整个测试集进行批量评估，生成性能指标对比表：

```bash
python test.py
```

**输出示例**:
```
======================================================================
模型评估结果汇总
======================================================================
+-------------+----------+--------+------------+-----------+
| Model       | Accuracy | mIoU   | Focal Loss | Dice Loss |
+=============+==========+========+============+===========+
| UNet        | 0.8523   | 0.6241 | 0.3456     | 0.2987    |
+-------------+----------+--------+------------+-----------+
| DeepLabV3   | 0.8687   | 0.6512 | 0.3124     | 0.2756    |
+-------------+----------+--------+------------+-----------+
| PSPNet      | 0.8456   | 0.6189 | 0.3567     | 0.3012    |
+-------------+----------+--------+------------+-----------+
======================================================================
```

**功能说明**:
- ✅ 自动加载三个模型的权重文件
- ✅ 对整个测试集进行批量预测
- ✅ 计算 Accuracy、mIoU、Focal Loss、Dice Loss 等指标
- ✅ 生成对比表格，便于模型选择


#### Step 3: 单图推理与可视化

使用 `inference.py` 对指定图片进行预测并生成分割结果可视化：

```bash
# 基本用法
python inference.py --model unet --weights ./runs/unet_final.pth --image path/to/image.jpg

# 带标签的完整评估（可选）
python inference.py `
    --model deeplabv3 `
    --weights ./runs/deeplabv3_final.pth `
    --image ./RescueNet/test/test-org-img/10782.jpg `
    --label ./RescueNet/test/test-label-img/10782_lab.png

# 指定输出文件名
python inference.py `
    --model pspnet `
    --weights ./runs/pspnet_final.pth `
    --image disaster_scene.jpg `
    --output my_result.png
```

**命令行参数**:
- `--model`: 模型类型（unet / pspnet / deeplabv3）
- `--weights`: 模型权重文件路径（必需）
- `--image`: 输入图像路径（必需）
- `--label`: 标签图像路径（可选，用于计算指标）
- `--output`: 输出文件名（默认 output.png）
- `--device`: 设备选择（cuda/cpu，默认自动）

**输出**:
- 彩色分割掩码图像保存到 `./output/` 目录
- 如果提供标签，还会打印评估指标（Accuracy、mIoU等）


#### Step 4: 运行应急响应智能体

```bash
# 单图：生成一份单图灾情报告
python agent_cli.py --single-turn --image-folder ./RescueNet/test/test-org-img/10782.jpg

# 批量：每张图分别生成一份独立报告
python agent_cli.py --single-turn --image-folder ./RescueNet/test/test-org-img

# 自定义问题
python agent_cli.py --single-turn \
    --image-folder ./RescueNet/test/test-org-img \
    --question "哪些区域的建筑物损坏最严重？应该优先派遣什么资源？" \
    --output critical_areas_report.html

```

报告以单张航拍图为分析单位，包含该图的受灾等级、损坏类别占比、道路阻塞情况、应急救援结论，以及位于右上角的语义分割图。目录输入仅批量执行单图逻辑，不再生成跨图汇总报告。

---

## 7. 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|---------|
| DeepSeek API不稳定 | 高 | 实现重试机制 + 备用LLM配置 |
| 模型权重文件缺失 | 中 | 启动时检查 + 明确错误提示 |
| GPU显存不足 | 中 | 串行处理模型 + 降低batch size |
| LLM输出格式错误 | 低 | 正则表达式容错 + 默认回退策略 |

---

## 8. 未来扩展方向

1. **多源数据融合**: 结合卫星图像、社交媒体文本、传感器数据
2. **时序演化分析**: 分析灾情随时间的变化趋势
3. **实时视频流处理**: 对接无人机实时画面
4. **多语言支持**: 国际化报告生成
5. **GIS集成**: 与地理信息系统对接，生成地图标注

---

**文档版本**: v1.0  
**最后更新**: 2026-06-19  
**作者**: RescueNet-Segmentation Team
