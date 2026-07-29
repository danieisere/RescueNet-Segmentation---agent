"""
通用命令行参数解析器
"""
import argparse


def get_train_parser():
    """获取训练脚本的参数解析器"""
    parser = argparse.ArgumentParser(description='语义分割训练脚本')
    
    parser.add_argument('--model', type=str, default='all',
                        choices=['unet', 'pspnet', 'deeplabv3', 'all'],
                        help='要训练的模型 (默认: all)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数 (默认: 10)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='批次大小 (默认: 2)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率 (默认: 1e-4)')
    parser.add_argument('--max-images', type=int, default=100,
                        help='最大训练图片数，None表示全部 (默认: 100)')
    parser.add_argument('--save-dir', type=str, default='./runs',
                        help='模型保存目录 (默认: ./runs)')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)，默认自动选择')
    
    return parser


def get_test_parser():
    """获取测试脚本的参数解析器"""
    parser = argparse.ArgumentParser(description='语义分割测试脚本')
    
    parser.add_argument('--model', type=str, default='all',
                        choices=['unet', 'pspnet', 'deeplabv3', 'all'],
                        help='要测试的模型 (默认: all)')
    parser.add_argument('--weights-dir', type=str, default='./runs',
                        help='模型权重目录 (默认: ./runs)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='批次大小 (默认: 2)')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)，默认自动选择')
    
    return parser


def parse_train_args(args=None):
    """解析训练参数"""
    parser = get_train_parser()
    return parser.parse_args(args)


def parse_test_args(args=None):
    """解析测试参数"""
    parser = get_test_parser()
    return parser.parse_args(args)
