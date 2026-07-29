"""
推理入口脚本
对单张图像进行预测和可视化
"""
import torch
import argparse

from models import get_model
from modules.visualization import predict_and_draw
from configs.dataset_config import TARGET_CLASSES


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='语义分割推理脚本')
    
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'pspnet', 'deeplabv3'],
                        help='模型类型 (默认: unet)')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--image', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--label', type=str, default=None,
                        help='标签图像路径（可选，用于计算指标）')
    parser.add_argument('--output', type=str, default='output.png',
                        help='输出文件名')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)，默认自动选择')
    
    return parser.parse_args()


def main():
    """主推理函数"""
    args = parse_args()
    
    # 选择设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    print(f"加载模型: {args.model}")
    print(f"权重文件: {args.weights}")
    print(f"输入图像: {args.image}")
    
    # 加载模型
    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print("模型加载成功！")
    
    # 进行预测和可视化
    result = predict_and_draw(
        model=model,
        image_path=args.image,
        device=device,
        model_name=args.model.upper(),
        label_path=args.label,
        color_mask_name=args.output,
        target_classes=TARGET_CLASSES
    )
    
    if result:
        print("\n评估结果:")
        for key, value in result.items():
            print(f"  {key}: {value:.4f}")
    
    print(f"\n预测结果已保存到 ./output/{args.output}")


if __name__ == "__main__":
    main()
