# RERA 快速开始

当前默认模式是多轮 LangGraph 会话。直接执行 `python agent_cli.py` 会创建会话并等待输入；`--chat` 可以省略。需要运行一次后退出，或者处理图片目录时，显式添加 `--single-turn`。

## 1. 准备环境

```bash
pip install -r requirements.txt
copy config.yaml.example config.yaml
```

编辑 `config.yaml`，填写大模型服务地址、模型和凭据。服务必须支持图片输入。准备三个分割权重后，确认它们位于 `runs/` 或 `ensemble.weight_dir` 配置的目录。

如果数据集尚未准备，先运行：

```bash
python verify_dataset.py
python train.py
```

## 1.1 浏览器灾情工作区

完成上面的配置与权重准备后，额外安装并启动网页：

```bash
# 只补充可选Web依赖，使用项目已有Python环境。
python -m pip install -r requirements-web.txt
# 直接启动网页，默认地址为 http://127.0.0.1:8000。
python web_app.py
```

打开 <http://127.0.0.1:8000>，推荐 Edge/Chrome：

1. 左侧“新建分析会话”创建以当前时间命名的会话，也可直接选图后发送，自动创建会话。
2. “选择本地文件夹”生成本地航拍图集；勾选后点击“添加到对话”，待发送附件可逐一移除。
3. 输入“只分割这几张图”或“生成完整 HTML 灾情综合报告”等需求后发送。浏览图集不上传，发送时只上传附件；每轮数量沿用 `vision.max_input_images`，默认最多 5 张，默认单图 20 MiB。
4. 中间查看真实执行阶段及回答；右侧查看原图/分割图/叠加图和共享图例，按轮次、图片或多图组筛选报告与数据。
5. 多图报告包含综合总览和分图章节，同时保留独立报告。名称沿用图片名，三张及以上使用前两张名称加 `etc_N_images_report`，后续报告不会覆盖历史版本。综合 HTML 内嵌图像，三图实测样本约 101 MB，首次预览和下载可能需要等待。
6. 继续补充“共享总资源”或明确点名某张图的现场信息。图片专属事实不复制到其他图片；无法定位的信息会先澄清。
7. 可关闭页面再进入历史会话；服务器继续已提交任务。本地目录因浏览器权限需要重新选择。后端重启导致的中断可重试原任务，已有阶段仍需校验才能复用。

左下设置面板与实际 YAML 联动；保存后重启后端生效。配置来源和运行中旧值均可查看，不显示密钥原文。请保留单个 Uvicorn worker；真实分析时不使用 `--reload`。首版不支持账号、PDF、地图拼接或任务强制取消。

## 2. 默认多轮会话

最简单的启动方式：

```bash
python agent_cli.py
```

带图片进入第一轮：

```bash
python agent_cli.py \
  --image-folder ./RescueNet/test/test-org-img/10794.jpg \
  --question "先统计各类别像素占比，不生成报告"
```

执行完首轮后，终端不会自动结束，会继续等待下一轮。比如可以继续输入：

```text
补充现场有10名应急专家将进场
请根据前面的结果给出应急救援方案
好，把该方案生成为 HTML 报告
```

输入 `/exit` 或 `退出` 会保存并结束当前终端；之后使用显示的 ID 恢复：

```bash
python agent_cli.py --session-id 20261031_165539
```

会话命令：

```text
/help                 查看帮助
/image 图片路径        切换当前图片
/sessions             列出历史会话
/new                  新建会话并保留旧会话
/exit                 保存并退出
```

会话目录默认是 `message/<YYYYMMDD_HHMMSS>/`。每个普通输入对应一轮，关闭终端不会删除消息、摘要、事实或历史产物。

## 3. 单轮和批量执行

只需要一次结果时：

```bash
# 默认给出救援建议，不生成报告
python agent_cli.py --single-turn \
  --image-folder ./RescueNet/test/test-org-img/10801.jpg

# 只统计
python agent_cli.py --single-turn \
  --image-folder ./RescueNet/test/test-org-img/10794.jpg \
  --question "只统计类别像素和占比，不要报告"

# 只做三图视觉解释
python agent_cli.py --single-turn \
  --image-folder ./RescueNet/test/test-org-img/10836.jpg \
  --question "结合三图解释道路损坏分布，不要救援建议，不要报告"

# 显式生成报告
python agent_cli.py --single-turn \
  --image-folder ./RescueNet/test/test-org-img/10808.jpg \
  --output assessment.html

# 处理整个目录，每张图片单独归档
python agent_cli.py --single-turn \
  --image-folder ./RescueNet/test/test-org-img
```

单轮/批量默认使用 LangGraph。ReAct 作为兼容引擎保留：

```bash
python agent_cli.py --single-turn --engine react \
  --image-folder ./RescueNet/test/test-org-img/10838.jpg \
  --output assessment.html
```

`--output` 只能在单轮/批量模式使用。多轮会话中，应在自然语言消息中要求生成报告并指定 HTML 或 Markdown。

## 4. LangGraph 会执行什么

LLM 先把请求转换成受校验的目标，图路由根据真实依赖决定阶段：

| 用户请求 | 阶段 |
|---|---|
| 生成分割图或叠加图 | 分割 → 保存产物 |
| 统计类别像素和占比 | 分割 → 统计 → 保存产物 |
| 计算救援优先级 | 分割 → 统计 → 优先级 → 保存产物 |
| 结合三图解释 | 分割 → 统计 → 多模态解释 → 保存产物 |
| 给出救援建议 | 分割 → 统计 → 优先级 → 多模态救援分析 → 保存产物 |
| 生成完整报告 | 分割 → 统计 → 优先级 → 多模态救援分析 → 报告 → 保存产物 |

节点完成后通过 LangGraph 流式事件输出开始、结束、重试、阶段复用、产物保存和终态。事件中的摘要是公开状态说明，不是模型完整内部思维链。

## 5. 查看产物

单轮/批量产物位于：

```text
output/artifacts/<run_id>/items/<item_id>/
```

其中：

- `images/`：原图、彩色分割图、半透明叠加图；
- `data/mask.npy`：原始类别掩膜；
- `data/visualization.json`：图例和尺寸映射；
- `data/statistics.json`：统计结果；
- `data/priority.json`：规则优先级；
- `data/analysis.json`：结构化综合分析；
- `report/`：用户明确要求时生成的 HTML/Markdown 报告；
- `manifest.json`、`events.jsonl`、`result.json`：来源、哈希、事件和最终状态。

多轮产物位于：

```text
message/<session_id>/turns/<turn_id>/artifacts/<run_id>/items/<item_id>/
```

历史轮次不会被新报告覆盖。后续轮次可以验证并复用仍然有效的分割、统计和优先级结果。

## 6. 上下文压缩

默认文本压缩阈值为 8000，目标长度为 6000。压缩时：

- 原始消息继续完整保留；
- 当前请求、有效事实和相关方案来源必须保留；
- 较早对话按需压缩，长消息可以分段整理；
- 压缩失败时保留旧摘要和原始消息；
- 终端输出压缩开始、重试和完成事件。

8000 是应用层预算，不等于服务商上下文窗口。模型窗口可以通过 `conversation.model_context_window` 配置；没有本地 tokenizer 缓存时，系统会明确使用保守估算，不会隐式联网下载。

## 7. 退出码

单轮/批量模式：

- `0`：所有图片完成；
- `1`：至少一张图片执行失败；
- `2`：只有澄清或不支持请求，没有执行失败。

多轮模式会持续交互；单轮结果中的 `status` 和归档路径以程序输出为准。

## 8. 常见问题

**为什么默认命令没有立刻结束？**

因为默认模式是多轮会话。需要一次性运行时使用 `--single-turn`。

**为什么只问统计却没有报告？**

报告是独立目标。统计请求不会自动生成报告；请在同一会话中继续要求生成报告，或在单轮模式中使用 `--output`。

**为什么报告里有三张图？**

多模态分析同时读取原图、分割图和半透明叠加图。分割图和报告使用同一套图例，类别统计仍使用原始掩膜数组。

**报告生成失败后数据会丢失吗？**

不会。已经成功的阶段会保留在对应 item 中；失败轮次和失败原因也会归档，后续请求可以验证并复用有效阶段。

**如何查看完整图？**

查看根目录的 [LANGGRAPH_ARCHITECTURE.md](LANGGRAPH_ARCHITECTURE.md)、[LANGGRAPH_WORKFLOW.md](LANGGRAPH_WORKFLOW.md) 和 [MULTITURN_ARCHITECTURE.md](MULTITURN_ARCHITECTURE.md)。
