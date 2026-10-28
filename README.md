# RescueNet应急响应智能体 (RERA) - 设计文档

## 1. 项目概述

### 1.1 项目名称
**RescueNet Emergency Response Agent (RERA)**

### 1.2 项目定位
基于ReAct范式的多模态灾害应急响应智能体，融合深度学习分割模型与大语言模型（DeepSeek），实现从"图像感知"到"决策建议"的完整闭环。

### 1.3 核心价值
- **解决痛点**: 灾后人工评估效率低，黄金救援时间宝贵
- **技术创新**: LLM控制器 + Ensemble视觉感知 + 规则引擎决策
- **实用价值**: 自动生成结构化救援报告，提升灾情评估效率80%+

---

## 2. 系统架构

### 2.1 整体架构

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
| **LLM控制器** | 意图理解、任务规划、报告生成 | DeepSeek API (deepseek-chat) |
| **Ensemble Segment Tool** | 多模型融合分割，选择最优结果 | PyTorch + 置信度评分 |
| **Statistics Analyzer** | 统计各类别数量、面积占比 | NumPy/Pandas |
| **Priority Calculator** | 计算救援优先级评分 | 加权算法 |
| **Report Generator** | 生成HTML/PDF报告 | Jinja2模板引擎 |

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
Priority_Score = w1 × N_destroyed + w2 × N_major + w3 × N_minor + w4 × Road_Blocked_Ratio

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
1. A区域有15栋完全毁坏建筑...
2. 建议优先派遣救援队到...
```

---


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
python agent_cli.py --image-folder ./RescueNet/test/test-org-img/10782.jpg

# 批量：每张图分别生成一份独立报告
python agent_cli.py --image-folder ./RescueNet/test/test-org-img

# 自定义问题
python agent_cli.py \
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
