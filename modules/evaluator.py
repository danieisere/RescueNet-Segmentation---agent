import torch
import numpy as np
from modules.metrics import pixel_accuracy_filtered, compute_miou
from modules.loss import FocalLoss, DiceLoss, get_class_weights


def evaluate_model(model, dataloader, device, target_classes=[2, 3, 4, 5]):
    """
    评估模型性能
    
    Args:
        model: 模型对象
        dataloader: 数据加载器
        device: 设备（cuda/cpu）
        target_classes: 关注的类别
        
    Returns:
        dict: 包含各项评估指标的字典
    """
    model.eval()
    
    class_weights = get_class_weights().to(device)
    focal_loss_fn = FocalLoss(gamma=2.0, weight=class_weights, target_classes=target_classes)
    dice_loss_fn = DiceLoss(target_classes=target_classes)
    
    total_acc = 0.0
    total_miou = 0.0
    total_focal_loss = 0.0
    total_dice_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            
            # 计算损失
            focal_loss = focal_loss_fn(outputs, labels)
            dice_loss = dice_loss_fn(outputs, labels)
            
            # 获取预测结果
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # 计算指标
            for i in range(preds.shape[0]):
                acc = pixel_accuracy_filtered(preds[i], labels_np[i], include_classes=target_classes)
                miou = compute_miou(preds[i], labels_np[i], num_classes=11, include_classes=target_classes)
                
                total_acc += acc
                total_miou += miou
            
            total_focal_loss += focal_loss.item()
            total_dice_loss += dice_loss.item()
            num_batches += 1
    
    avg_acc = total_acc / (num_batches * dataloader.batch_size)
    avg_miou = total_miou / (num_batches * dataloader.batch_size)
    avg_focal_loss = total_focal_loss / num_batches
    avg_dice_loss = total_dice_loss / num_batches
    
    return {
        'accuracy': avg_acc,
        'miou': avg_miou,
        'focal_loss': avg_focal_loss,
        'dice_loss': avg_dice_loss
    }
