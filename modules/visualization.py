import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from modules.loss import FocalLoss, DiceLoss, get_class_weights
from modules.metrics import pixel_accuracy_filtered, compute_miou


# 可视化调色板
VISUALIZATION_PALETTE = {
    0: [0, 0, 0],  # 背景 - 黑色
    1: [0, 0, 255],  # 水 - 纯蓝色
    2: [150, 150, 150],  # 建筑物无损 - 浅灰
    3: [255, 255, 0],  # 建筑物轻微损坏 - 黄色
    4: [255, 165, 0],  # 建筑物严重损坏 - 橙色
    5: [255, 0, 0],  # 建筑物完全毁坏 - 红色
    6: [0, 255, 255],  # 车辆 - 青色
    7: [128, 64, 128],  # 道路畅通 - 紫灰色
    8: [64, 0, 128],  # 道路阻塞 - 深紫色
    9: [0, 128, 0],  # 树木 - 绿色
    10: [0, 191, 255],  # 水池 - 天蓝色
}


def predict_and_draw(model, image_path, device, model_name="", label_path=None, 
                     save_name=None, color_mask_name=None, target_classes=[2, 3, 4, 5]):
    """
    对单张图像进行预测并可视化结果
    
    Args:
        model: 模型对象
        image_path: 输入图像路径
        device: 设备（cuda/cpu）
        model_name: 模型名称
        label_path: 标签图像路径（可选，用于计算指标）
        save_name: 保存文件名
        color_mask_name: 彩色掩码保存文件名
        target_classes: 关注的类别
        
    Returns:
        dict: 包含评估指标的字典（如果有label_path）
    """
    model.eval()

    # 加载图像并预处理
    raw_image = cv2.imread(image_path)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(768, 768),
        A.Normalize(),
        ToTensorV2()
    ])
    augmented = transform(image=raw_image)
    img_tensor = augmented['image'].unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 输出预测的伪彩图
    color_label = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_id, color in VISUALIZATION_PALETTE.items():
        color_label[pred == class_id] = color

    plt.figure(figsize=(6, 6))
    plt.imshow(color_label)
    plt.title(f"{model_name} 预测结果的彩色标签图" if model_name else "预测结果的彩色标签图")
    plt.axis("off")
    plt.show()
    
    # 保存彩色掩码
    if color_mask_name:
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, color_mask_name),
                    cv2.cvtColor(color_label, cv2.COLOR_RGB2BGR))

    # 如果提供了标签路径，计算评估指标
    result = {}
    if label_path is not None:
        class_weights = get_class_weights().to(device)
        focal_loss_fn = FocalLoss(gamma=2.0, weight=class_weights, target_classes=target_classes)
        dice_loss_fn = DiceLoss(target_classes=target_classes)
        
        # 加载并Resize标签图
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        resized_label = A.Resize(768, 768)(image=label_img)['image']

        # 转成tensor并与output对齐
        label_tensor = torch.from_numpy(resized_label).long().unsqueeze(0).to(device)

        # 计算两类loss
        focal_loss = focal_loss_fn(output, label_tensor)
        dice_loss = dice_loss_fn(output, label_tensor)

        # 输出性能指标
        acc = pixel_accuracy_filtered(pred, resized_label, include_classes=target_classes)
        miou = compute_miou(pred, resized_label, num_classes=11, include_classes=target_classes)

        print(f"Pixel Accuracy: {acc:.4f}")
        print(f"平均IoU（mIoU）: {miou:.4f}")
        print(f"Focal Loss: {focal_loss.item():.4f}")
        print(f"Dice Loss: {dice_loss.item():.4f}")

        result = {
            'accuracy': acc,
            'miou': miou,
            'Focal Loss': focal_loss.item(),
            'Dice Loss': dice_loss.item(),
        }
    
    return result
