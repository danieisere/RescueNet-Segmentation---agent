import numpy as np


def pixel_accuracy_filtered(pred, label, include_classes=[2, 3, 4, 5]):
    """
    仅在指定类别上计算像素准确率
    
    Args:
        pred: numpy array, shape [H, W]，预测结果
        label: numpy array, shape [H, W]，真实标签
        include_classes: list[int]，只统计这些类别的像素
        
    Returns:
        float: 过滤后的像素准确率
    """
    mask = np.isin(label, include_classes)  # 只看指定类别的区域
    correct = (pred[mask] == label[mask]).sum()
    total = mask.sum()
    return correct / total if total > 0 else 0.0


def compute_miou(pred, label, num_classes=11, include_classes=None):
    """
    计算 mean IoU，可选指定参与计算的类别
    
    Args:
        pred: 预测的标签图，shape = [H, W]
        label: 真实标签图，shape = [H, W]
        num_classes: 数据集中总类别数（默认11）
        include_classes: 指定计算哪些类别的 IoU，例如 [2, 3, 4, 5] 表示只计算建筑物类
        
    Returns:
        float: 平均 IoU
    """
    ious = []
    for cls in range(num_classes):
        if include_classes is not None and cls not in include_classes:
            continue  # 跳过不感兴趣的类别

        pred_inds = (pred == cls)
        label_inds = (label == cls)
        intersection = np.logical_and(pred_inds, label_inds).sum()
        union = np.logical_or(pred_inds, label_inds).sum()

        if union == 0:
            continue  # 如果没有这个类的像素，就跳过

        iou = intersection / union
        ious.append(iou)

    return np.mean(ious) if ious else 0.0
