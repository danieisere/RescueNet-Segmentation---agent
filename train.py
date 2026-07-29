"""
训练入口脚本
支持多模型对比训练
"""
import os
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from models import get_model
from modules.datasets import SegmentationDataset
from modules.trainer import train
from configs.dataset_config import DATASET_CONFIG, DATALOADER_CONFIG
from configs.training_config import TRAINING_CONFIG, SAVE_CONFIG


def main():
    """主训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    train_dataset = SegmentationDataset(
        img_dir=DATASET_CONFIG['train_img_dir'],
        mask_dir=DATASET_CONFIG['train_mask_dir'],
        max_images=DATALOADER_CONFIG['max_images']
    )
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=DATALOADER_CONFIG['batch_size'],
        shuffle=True,
        num_workers=DATALOADER_CONFIG['num_workers'],
        pin_memory=DATALOADER_CONFIG['pin_memory'],
        drop_last=DATALOADER_CONFIG.get('drop_last', True)
    )

    # 定义要训练的模型
    models_dict = {
        "UNet": get_model('unet').to(device),
        "DeepLabV3": get_model('deeplabv3').to(device),
        "PSPNet": get_model('pspnet').to(device)
    }

    # 存储各模型的训练损失
    all_losses = {}
    
    # 确保保存目录存在
    os.makedirs(SAVE_CONFIG['model_save_dir'], exist_ok=True)

    # 训练每个模型
    for model_name, model in models_dict.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print(f"{'='*50}")
        
        losses = train(
            model=model,
            dataloader=dataloader,
            device=device,
            epochs=TRAINING_CONFIG['epochs'],
            save_path=SAVE_CONFIG['model_save_dir'],
            model_name=model_name.lower(),
            target_classes=TRAINING_CONFIG['target_classes'],
            lr=TRAINING_CONFIG['learning_rate'],
            save_interval=SAVE_CONFIG.get('save_interval')  # 添加 save_interval 参数
        )
        
        # 保存最终模型
        final_model_path = os.path.join(
            SAVE_CONFIG['model_save_dir'], 
            f"{model_name}_final.pth"
        )
        torch.save(model.state_dict(), final_model_path)
        print(f"{model_name} training completed and model saved to {final_model_path}!")
        
        all_losses[model_name] = losses['epoch_losses']

    # 绘制损失曲线
    plot_loss_curves(all_losses)


def plot_loss_curves(all_losses):
    """绘制训练损失曲线"""
    epochs = list(range(1, len(list(all_losses.values())[0]) + 1))
    
    plt.figure(figsize=(10, 6))
    
    colors = {'UNet': 'green', 'DeepLabV3': 'blue', 'PSPNet': 'red'}
    for model_name, losses in all_losses.items():
        plt.plot(epochs, losses, 'o-', color=colors.get(model_name, 'black'), label=model_name)

    plt.title('Training Loss Curve Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    
    # 保存图像
    os.makedirs('./output', exist_ok=True)
    plt.savefig("./output/损失函数曲线.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("损失曲线已保存到 ./output/损失函数曲线.png")


if __name__ == "__main__":
    main()
