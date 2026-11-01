# 🚀 RescueNet应急响应智能体 - 快速启动指南

默认多轮对话：运行 `python agent_cli.py` 即可，`--chat` 可以省略。需要执行完一次任务就退出，或批量处理目录时，添加 `--single-turn`。

当前默认引擎为 **LangGraph**。完整执行图与限制见 [LANGGRAPH_ARCHITECTURE.md](LANGGRAPH_ARCHITECTURE.md)。默认问题生成救援建议和独立数据产物；需要报告时显式给出 `--output name.html` / `name.md` 或在问题中要求完整报告。

```bash
# 分割 / 统计 / 优先级 / 视觉解释按需执行
python agent_cli.py --single-turn --image-folder ./RescueNet/test/test-org-img/10807.jpg --question "只要分割图和叠加图，不要报告"
python agent_cli.py --single-turn --image-folder ./RescueNet/test/test-org-img/10794.jpg --question "只统计类别像素和占比，不要报告"
python agent_cli.py --single-turn --image-folder ./RescueNet/test/test-org-img/10801.jpg --question "只计算救援优先级分数，不要建议和报告"
python agent_cli.py --single-turn --image-folder ./RescueNet/test/test-org-img/10836.jpg --question "结合三图解释道路分布，不要优先级和救援建议，不要报告"

# 完整报告 / 兼容引擎
python agent_cli.py --single-turn --image-folder ./RescueNet/test/test-org-img/10808.jpg --output assessment.html
python agent_cli.py --single-turn --engine react --image-folder ./RescueNet/test/test-org-img/10838.jpg --output assessment.md
```

进度输出示例（节点名来自实际图）：`validate → classify → segmentation → save_artifacts → statistics → ... → finalize`。每个长耗时节点先输出 `node_start`，不要求逐 token 输出。失败走统一终态并保留上游；需要澄清时结束本轮，修改问题重新运行。

新产物路径为 `output/artifacts/<run_id>/items/<item_id>/`：三图在 `images/`，真实掩膜和独立数据在 `data/`，按需报告在 `report/`。run/item 下的 `manifest.json`、`request.json`、`result.json`、`events.jsonl` 可供复核。旧 reports 已归档到 `output/artifacts/legacy_reports/`，下文旧 ReAct 日志只用于说明兼容引擎。

Windows CLI 在未显式配置 `MKL_THREADING_LAYER` 时默认使用 `SEQUENTIAL`，避免本地 NumPy/MKL 与 PyTorch 携带的 Intel OpenMP 冲突。该设置只影响本次 CLI 进程的 MKL 线程层；不绕过 OpenMP 检查，也不改动 CUDA 设备选择。直接从其他 Python 入口调用服务时，调用方需在导入数值库前处理自己的运行环境。

## 多轮终端对话

```bash
# 新建会话并开始交流；不提供图片也可先进行文字讨论
python agent_cli.py --image-folder ./RescueNet/test/test-org-img/10794.jpg --question "先统计类别占比"

# 恢复时替换为实际显示的秒级 ID
python agent_cli.py --session-id 20261031_153042
python agent_cli.py --list-sessions
```

同一终端反复输入属于同一会话。`/exit` 或 `退出` 保存并暂停，`/new` 创建新会话，`/image 图片路径` 切换当前图片，`/help` 查看命令。会话目录为根目录 `message/<session_id>/`，其中 `turns/` 包含各轮记录和新分析成果；`memory.sqlite` 是记忆权威来源，JSON/JSONL 便于直接阅读。单轮输出继续使用原 `output/artifacts/`。可通过 `--message-root` 或 `conversation.message_root` 覆盖会话位置。

支持事实补充、更正、关注重点和按需报告；“不是8人，是12人，再生成报告”先更新事实再生成。会话模式报告格式通过文字指定 HTML/Markdown，不使用单轮的 `--output`。详细架构、原始消息保留、压缩和恢复边界见 [MULTITURN_ARCHITECTURE.md](MULTITURN_ARCHITECTURE.md)。

历史超过应用的 8000 文本 token 预算时自动压缩并继续当前请求，不需要手动整理或新建会话。终端显示 `compression_start` / `compression_end`；原始消息完整保留，相关方案及来源会传入报告。近期 4 轮是优先保留而非不可压缩；默认压缩目标为 6000。UTF-8 保守估算可能提前触发，不能据此认定服务商模型窗口已满。

## ⚡ 5分钟快速开始

### Step 1: 安装依赖（1分钟）

```bash
pip install -r requirements.txt
```

或者使用requirements.txt：

```bash
pip install -r requirements.txt
```

---

### Step 2: 配置 LLM API Key（1分钟）

1. 从 `config.yaml` 中配置的 LLM 服务商获取 API Key
2. 确认 `llm.api_key_env` 是要使用的环境变量名
3. 复制配置和环境变量模板：

```bash
cp config.yaml.example config.yaml
cp .env.example .env
```

编辑 `.env`，填入与 `config.yaml` 中 `llm.api_key_env` 对应的 Key：

```dotenv
CODEXCN_API_KEY=your_codexcn_api_key_here
```

> ⚠️ `.env` 和 `config.yaml` 均已 gitignore，不会被提交到 GitHub。

---

### Step 3: 准备并运行模型

**Step 3.1: 下载数据集并训练模型（约2-3小时）**

> ⚠️ RescueNet 数据集不包含在本仓库中，请先从 [BinaLab/RescueNet](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation) 下载，并按以下结构放置：
> ```
> RescueNet/
> ├── train/train-org-img/   # 训练原始图像 (.jpg)
> ├── train/train-label-img/ # 训练标签掩码 (_lab.png)
> ├── test/test-org-img/     # 测试原始图像 (.jpg)
> └── test/test-label-img/   # 测试标签掩码 (_lab.png)
> ```

```bash
# 验证数据集结构
python verify_dataset.py

# 训练三个模型
python train.py
```

等待训练完成，确认以下文件存在：
```bash
ls runs/unet_final.pth runs/pspnet_final.pth runs/deeplabv3_final.pth
```

**Step 3.2: 运行Agent**

```bash
python agent_cli.py --single-turn --image-folder ./RescueNet/test/test-org-img
```

---

## 📋 完整命令参考

### 基本用法

```bash
# 单图：默认生成救援建议并保存独立产物
python agent_cli.py --single-turn --image-folder ./RescueNet/test/test-org-img/10782.jpg

# 批量：对目录内每张图分别执行独立单图流程
python agent_cli.py --single-turn --image-folder ./RescueNet/test/test-org-img

# 自定义问题
python agent_cli.py --single-turn \
    --image-folder ./images \
    --question "哪些区域建筑物损坏最严重？"

# 指定报告文件名；批量时会自动追加图片名避免覆盖
python agent_cli.py --single-turn \
    --image-folder ./images \
    --output my_disaster_report.html

# 跳过前置检查（不推荐）
python agent_cli.py --single-turn \
    --image-folder ./images \
    --skip-check
```

---

## 🎬 预期输出示例

### 输出示例

```
🔍 检查Agent运行前置条件...
============================================================
✅ 模型权重文件齐全
✅ DeepSeek API Key已配置
============================================================
✅ 所有前置条件满足，可以启动Agent！

📁 找到 1 张图像

🔧 初始化Agent工具...
✅ 已加载 4 个工具

============================================================
📝 用户问题: 分析这批灾区的损坏情况并生成救援建议
============================================================

🤖 Agent开始执行ReAct循环...
============================================================

[Step 1/10]
💭 Thought: 我需要先对图像进行分割处理
🔧 Action: ensemble_segment
📊 Observation: {"best_model": "deeplabv3", "scores": {...}}...

[Step 2/10]
💭 Thought: 现在需要统计各类别数量
🔧 Action: analyze_statistics
📊 Observation: {"per_image_stats": [...], "global_stats": {...}}...

[Step 3/10]
💭 Thought: 根据统计数据计算救援优先级
🔧 Action: calculate_priority
📊 Observation: {"priority_zones": [...], "statistics": {...}}...

[Step 4/10]
💭 Thought: 最后生成应急救援报告
🔧 Action: generate_report
📊 Observation: {"report_path": "./output/artifacts/<run_id>/items/<item_id>/report/emergency_report_...", ...}...

[Step 5/10]
💭 Thought: 任务已完成，可以给出最终回答
✅ Final Answer received

根据分析，发现以下紧急情况：
- 预测为完全毁坏建筑的像素占比为 X%
- 预测为严重损坏建筑的像素占比为 Y%
- 道路阻塞率Z%

建议优先派遣救援队到...

============================================================
📋 Final Answer:
============================================================
[LLM生成的完整回答]

✅ 报告已保存到: ./output/artifacts/<run_id>/items/<item_id>/report/emergency_report_20260619_234516.html
💡 提示: 在浏览器中打开查看完整报告

============================================================
🎉 Agent执行完成！
============================================================
```

---

## ❓ 常见问题

### Q1: 提示"缺少模型权重文件"？

**A**: 运行训练脚本生成所需权重：
```bash
python train.py
```

> 每份 HTML 报告只分析一张图像，右上角包含该图对应的语义分割图和关键类别图例。批量模式不会生成跨图汇总报告。

---

### Q2: DeepSeek API调用失败？

**A**: 检查以下几点：
1. API Key是否正确配置
2. 网络连接是否正常
3. API余额是否充足

---

### Q3: 如何查看生成的报告？

**A**: 报告仅在请求时生成，保存在 `./output/artifacts/<run_id>/items/<item_id>/report/`，直接打开 HTML 文件即可。独立分析 JSON 在同一 item 的 `data/analysis.json`，不依赖是否生成报告。

---

### Q4: 想调整优先级权重？

**A**: 编辑 `config.yaml` 中的 `priority_weights` 配置：

```yaml
priority_weights:
  destroyed: 0.6     # 调整这个值
  major: 0.3
  minor: 0.1
  road_blocked: 0.2
```

---

## 🎯 下一步

1. **阅读设计文档**: 查看 `AGENT_DESIGN.md` 了解完整架构
2. **自定义问题**: 尝试不同的用户查询，观察Agent的推理过程
3. **扩展功能**: 添加新的Tool或调整现有算法
4. **部署Web界面**: 实现Streamlit交互式界面（可选）

---

**祝你使用愉快！** 🚀

如有问题，请查看 `AGENT_DESIGN.md` 或提交Issue。
