# 🚀 RescueNet应急响应智能体 - 快速启动指南

## ⚡ 5分钟快速开始

### Step 1: 安装依赖（1分钟）

```bash
pip install torch torchvision opencv-python albumentations matplotlib tqdm tabulate Pillow openai jinja2 weasyprint
```

或者使用requirements.txt：

```bash
pip install -r requirements.txt
```

---

### Step 2: 配置DeepSeek API Key（1分钟）

1. 注册DeepSeek账号: https://platform.deepseek.com/
2. 获取API Key
3. 复制配置模板并填入 Key:

```bash
cp config.yaml.example config.yaml
```

编辑 `config.yaml`:

```yaml
llm:
  api_key: "sk-your-deepseek-api-key-here"  # ← 替换为你的Key
```

> ⚠️ `config.yaml` 已 gitignore，不会被提交到 GitHub。

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
python agent_cli.py --image-folder ./RescueNet/test/test-org-img
```

---

## 📋 完整命令参考

### 基本用法

```bash
# 默认问题
python agent_cli.py --image-folder ./RescueNet/test/test-org-img

# 自定义问题
python agent_cli.py \
    --image-folder ./images \
    --question "哪些区域建筑物损坏最严重？"

# 指定输出文件名
python agent_cli.py \
    --image-folder ./images \
    --output my_disaster_report.html

# 跳过前置检查（不推荐）
python agent_cli.py \
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
📊 Observation: {"report_path": "./output/reports/emergency_report_...", ...}...

[Step 5/10]
💭 Thought: 任务已完成，可以给出最终回答
✅ Final Answer received

根据分析，发现以下紧急情况：
- 检测到X栋完全毁坏建筑
- Y栋严重损坏建筑
- 道路阻塞率Z%

建议优先派遣救援队到...

============================================================
📋 Final Answer:
============================================================
[LLM生成的完整回答]

✅ 报告已保存到: ./output/reports/emergency_report_20260619_234516.html
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

---

### Q2: DeepSeek API调用失败？

**A**: 检查以下几点：
1. API Key是否正确配置
2. 网络连接是否正常
3. API余额是否充足

---

### Q3: 如何查看生成的报告？

**A**: 报告保存在 `./output/reports/` 目录下，直接用浏览器打开HTML文件即可。

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
