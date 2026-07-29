import cv2
import os
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(Dataset):
    """
    图像语义分割数据集类
    
    数据流程：
    文件名列表 imgs → 读取 image、推导出 mask_name → 
    OpenCV 读取 RGB 图像 + 灰度标签 → Albumentations 同步增强处理 → 
    转换为 Tensor 格式 → 返回 image, mask（一个训练样本）
    
    RescueNet数据集命名规则: xxx.jpg → xxx_lab.png
    """
    def __init__(self, img_dir, mask_dir, max_images=None, transform=None):
        """
        Args:
            img_dir: 图像文件夹路径
            mask_dir: 标签（掩码）文件夹路径
            max_images: 最多读取的图片数量，None表示全部读取
            transform: 自定义数据增强变换，None则使用默认变换
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        # 获取 img_dir 目录下所有的图像文件（后缀是 .jpg 或 .png）
        self.imgs = [img for img in os.listdir(img_dir) if img.endswith(('.jpg', '.png'))]
        
        # 按文件名排序以保证可重复性
        self.imgs.sort()

        # 只选取前max_images张图片
        if max_images is not None:
            self.imgs = self.imgs[:max_images]

        # 默认数据增强流水线
        if transform is None:
            self.transform = A.Compose([
                A.Resize(1024, 1024),
                A.HorizontalFlip(p=0.5),      # 水平翻转
                A.VerticalFlip(p=0.5),        # 垂直翻转
                A.RandomRotate90(p=0.5),      # 随机旋转90度
                A.ElasticTransform(p=0.3),    # 弹性变换
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            self.transform = transform

    def __len__(self):
        """返回数据集的长度（图片数量），用于 DataLoader 迭代"""
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        获取第 idx 个图像-标签对
        
        Returns:
            image: float32 Tensor (3, H, W)
            mask: int64 Tensor (H, W)，代表类别索引
        """
        # 获取图像文件名和对应的完整路径
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 从图像名推导出标签名，比如 0001.jpg → 0001_lab.png
        if img_name.endswith('.jpg'):
            mask_name = img_name.replace('.jpg', '_lab.png')
        elif img_name.endswith('.png'):
            mask_name = img_name.replace('.png', '_lab.png')
        else:
            raise ValueError(f"Unsupported file format: {img_name}")

        mask_path = os.path.join(self.mask_dir, mask_name)

        # 使用 OpenCV 读取图像和掩码
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 如果图像或标签没读成功，抛出异常
        if image is None:
            raise ValueError(f"Failed to read image {img_path}.")
        if mask is None:
            raise ValueError(f"Failed to read mask {mask_path}.")

        # 应用数据增强并转换为Tensor
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].long()

        return image, mask
