from torch.utils.data import DataLoader
from modules.datasets import SegmentationDataset


def create_dataloader(img_dir, mask_dir, batch_size=2, max_images=None, 
                      shuffle=True, num_workers=6, pin_memory=True):
    """
    创建数据加载器的工厂函数
    
    Args:
        img_dir: 图像文件夹路径
        mask_dir: 标签文件夹路径
        batch_size: 批次大小
        max_images: 最大图片数量，None表示全部
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
        pin_memory: 是否使用锁页内存
        
    Returns:
        DataLoader对象
    """
    dataset = SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        max_images=max_images
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader
