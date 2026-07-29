"""数据集配置参数"""
import os

# 获取项目根目录（假设此文件在 configs/ 目录下）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# RescueNet数据集路径配置（使用绝对路径，避免相对路径问题）
DATASET_CONFIG = {
    'train_img_dir': os.path.join(PROJECT_ROOT, 'RescueNet', 'train', 'train-org-img'),
    'train_mask_dir': os.path.join(PROJECT_ROOT, 'RescueNet', 'train', 'train-label-img'),
    'test_img_dir': os.path.join(PROJECT_ROOT, 'RescueNet', 'test', 'test-org-img'),
    'test_mask_dir': os.path.join(PROJECT_ROOT, 'RescueNet', 'test', 'test-label-img'),
}

# 数据加载配置
DATALOADER_CONFIG = {
    'batch_size': 2,
    'num_workers': 6,
    'pin_memory': True,
    'max_images': 2000,  # None表示使用全部数据，改为 None 以使用所有训练样本
    'drop_last': True,  # 丢弃最后不足一个批次的数据，防止 BatchNorm 错误
}

# 图像预处理配置
IMAGE_CONFIG = {
    'resize_height': 1024,
    'resize_width': 1024,
    'num_classes': 11,
}

# 类别定义（RescueNet）
CLASS_NAMES = {
    0: "背景",
    1: "水",
    2: "建筑物无损",
    3: "建筑物轻微损坏",
    4: "建筑物严重损坏",
    5: "建筑物完全毁坏",
    6: "车辆",
    7: "道路畅通",
    8: "道路阻塞",
    9: "树木",
    10: "水池"
}

# 评估时关注的类别（建筑物相关）
TARGET_CLASSES = [2, 3, 4, 5]
