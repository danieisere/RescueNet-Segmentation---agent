import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss，用于处理类别不平衡问题"""
    def __init__(self, gamma=2.0, weight=None, target_classes=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.target_classes = target_classes

    def forward(self, input, target):
        N, C, H, W = input.shape
        input = input.permute(0, 2, 3, 1).reshape(-1, C)  # [N*H*W, C]
        target = target.view(-1)  # [N*H*W]

        if self.target_classes is not None:
            # 只保留目标标签
            valid_mask = torch.zeros_like(target, dtype=torch.bool)
            for cls in self.target_classes:
                valid_mask |= (target == cls)
            
            input = input[valid_mask]
            target = target[valid_mask]
        
        # 检查是否有有效样本
        if target.numel() == 0:
            return torch.tensor(0.0, device=input.device)

        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)

        # 数值稳定性保护：clamp 避免极值
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
        logpt = torch.clamp(logpt, min=-50, max=0)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.weight is not None:
            wt = self.weight.gather(0, target)
            loss *= wt
        
        # 限制损失值范围，防止爆炸
        loss = torch.clamp(loss, min=0, max=100)
        return loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss，用于分割任务"""
    def __init__(self, smooth=1e-5, target_classes=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.target_classes = target_classes

    def forward(self, input, target):
        input = F.softmax(input, dim=1)  # [B, C, H, W]
        num_classes = input.size(1)
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        if self.target_classes is not None:
            # 只保留目标类的通道
            input = input[:, self.target_classes, :, :]
            target_one_hot = target_one_hot[:, self.target_classes, :, :]

        dims = (0, 2, 3)
        intersection = torch.sum(input * target_one_hot, dims)
        union = torch.sum(input + target_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss，Focal Loss和Dice Loss的泛化"""
    def __init__(self, alpha=0.6, beta=0.4, smooth=1e-6, target_classes=None):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.target_classes = target_classes

    def forward(self, input, target):
        input = F.softmax(input, dim=1)  # [B, C, H, W]
        num_classes = input.shape[1]

        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        if self.target_classes is not None:
            input = input[:, self.target_classes, :, :]
            target_one_hot = target_one_hot[:, self.target_classes, :, :]

        TP = (input * target_one_hot).sum(dim=(2, 3))
        FP = ((1 - target_one_hot) * input).sum(dim=(2, 3))
        FN = (target_one_hot * (1 - input)).sum(dim=(2, 3))

        # 数值稳定性保护：确保分母不为零且结果合理
        denominator = TP + self.alpha * FP + self.beta * FN + self.smooth
        denominator = torch.clamp(denominator, min=1e-7)
        
        tversky = (TP + self.smooth) / denominator
        
        # clamp 防止极端值
        tversky = torch.clamp(tversky, min=0, max=1.0)
        
        loss = 1 - tversky.mean()
        # 限制损失值范围
        loss = torch.clamp(loss, min=0, max=100)
        return loss


def get_class_weights():
    """
    计算类别权重，用于处理类别不平衡
    
    Returns:
        torch.Tensor: 类别权重张量
    """
    pixel_counts = np.array([
        2.26798943e+10, 3.50981388e+09, 1.14351218e+09, 1.13662843e+09,
        7.26309268e+08, 6.22711602e+08, 1.43126533e+08, 2.99246963e+09,
        6.85492293e+08, 9.52374313e+09, 2.45137680e+07
    ])
    pixel_counts = np.maximum(pixel_counts, 1)
    class_weights = 1.0 / np.log(pixel_counts + 1)  # 对数平滑
    class_weights = class_weights / class_weights.max() * 10
    return torch.tensor(class_weights, dtype=torch.float32)
