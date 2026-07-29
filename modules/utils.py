import os
import torch
import numpy as np


def set_seed(seed=42):
    """
    设置随机种子以保证实验可重复性
    
    Args:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.backends.cudnn.deterministic:
        torch.backends.cudnn.benchmark = False


def create_directories(dirs):
    """
    创建目录（如果不存在）
    
    Args:
        dirs: 目录路径列表
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def get_device():
    """
    获取可用的训练设备
    
    Returns:
        torch.device: CUDA或CPU设备
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
