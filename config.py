"""
全局配置（兼容旧代码）
从configs目录导入所有配置
"""
from configs.dataset_config import *
from configs.model_config import *
from configs.training_config import *

# 导出所有配置，保持向后兼容
__all__ = [
    'DATASET_CONFIG', 'DATALOADER_CONFIG', 'IMAGE_CONFIG', 
    'CLASS_NAMES', 'TARGET_CLASSES',
    'SUPPORTED_MODELS', 'MODEL_CONFIG', 'UNET_CONFIG', 
    'PSPNET_CONFIG', 'DEEPLABV3_CONFIG',
    'TRAINING_CONFIG', 'LOSS_CONFIG', 'OPTIMIZER_CONFIG', 'SAVE_CONFIG'
]
