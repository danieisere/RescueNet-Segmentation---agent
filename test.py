"""
测试/评估入口脚本
对多个模型进行评估并输出对比结果
"""
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader

from models import get_model
from modules.datasets import SegmentationDataset
from modules.evaluator import evaluate_model
from configs.dataset_config import DATASET_CONFIG, DATALOADER_CONFIG, TARGET_CLASSES


def main():
    """主测试函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 定义要测试的模型
    models_dict = {
        "UNet": get_model('unet').to(device),
        "DeepLabV3": get_model('deeplabv3').to(device),
        "PSPNet": get_model('pspnet').to(device)
    }

    # 加载模型权重
    model_weights = {
        "UNet": "./runs/unet_final.pth",
        "DeepLabV3": "./runs/deeplabv3_final.pth",
        "PSPNet": "./runs/pspnet_final.pth"
    }

    # 创建测试数据加载器
    test_dataset = SegmentationDataset(
        img_dir=DATASET_CONFIG['test_img_dir'],
        mask_dir=DATASET_CONFIG['test_mask_dir'],
        max_images=None  # 使用全部测试数据
    )
    
    dataloader = DataLoader(
        test_dataset,
        batch_size=DATALOADER_CONFIG['batch_size'],
        shuffle=False,
        num_workers=DATALOADER_CONFIG['num_workers'],
        pin_memory=DATALOADER_CONFIG['pin_memory']
    )

    # 存储各个模型的评估结果
    evaluation_results = []

    # 评估每个模型
    for model_name, model in models_dict.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}...")
        print(f"{'='*50}")
        
        # 加载模型权重
        weight_path = model_weights[model_name]
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device))
            print(f"成功加载模型权重: {weight_path}")
        except FileNotFoundError:
            print(f"警告: 未找到模型权重文件 {weight_path}，使用随机初始化的模型")
        
        # 评估模型
        metrics = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            target_classes=TARGET_CLASSES
        )
        
        # 记录评估结果
        evaluation_results.append({
            "Model": model_name,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "mIoU": f"{metrics['miou']:.4f}",
            "Focal Loss": f"{metrics['focal_loss']:.4f}",
            "Dice Loss": f"{metrics['dice_loss']:.4f}"
        })
        
        print(f"{model_name} 评估完成！")

    # 打印表格形式的评估结果
    print("\n" + "="*70)
    print("模型评估结果汇总")
    print("="*70)
    headers = ["Model", "Accuracy", "mIoU", "Focal Loss", "Dice Loss"]
    table = [[r["Model"], r["Accuracy"], r["mIoU"], r["Focal Loss"], r["Dice Loss"]] 
             for r in evaluation_results]
    print(tabulate(table, headers=headers, tablefmt="grid"))
    print("="*70)


if __name__ == "__main__":
    main()
