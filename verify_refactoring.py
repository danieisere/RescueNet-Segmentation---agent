"""
项目重构验证脚本
用于测试新架构的各个模块是否可以正常导入和工作
"""
import sys
import torch


def print_section(title):
    """打印分隔标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_imports():
    """测试所有模块的导入"""
    print_section("1. 测试模块导入")
    
    try:
        # 测试models包
        from models import get_model, UNet, PSPNet, DeepLabV3_Custom
        print("✅ models包导入成功")
        
        # 测试layers
        from models.layers import conv_block, up_conv, SEBlock, CustomPyramidModule, PSPModule, ASPP
        print("✅ models.layers导入成功")
        
        # 测试modules包
        from modules import SegmentationDataset, FocalLoss, DiceLoss, TverskyLoss
        from modules import train, evaluate_model, predict_and_draw
        from modules import pixel_accuracy_filtered, compute_miou
        print("✅ modules包导入成功")
        
        # 测试configs包
        from configs.dataset_config import DATASET_CONFIG, DATALOADER_CONFIG
        from configs.model_config import MODEL_CONFIG
        from configs.training_config import TRAINING_CONFIG
        print("✅ configs包导入成功")
        
        # 测试packages
        from packages.argument_parser import get_train_parser, get_test_parser
        print("✅ packages包导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """测试模型创建"""
    print_section("2. 测试模型创建")
    
    try:
        from models import get_model
        
        # 测试UNet
        unet = get_model('unet', num_classes=11)
        print(f"✅ UNet创建成功 - 参数量: {sum(p.numel() for p in unet.parameters()):,}")
        
        # 测试PSPNet
        pspnet = get_model('pspnet', num_classes=11)
        print(f"✅ PSPNet创建成功 - 参数量: {sum(p.numel() for p in pspnet.parameters()):,}")
        
        # 测试DeepLabV3
        deeplab = get_model('deeplabv3', num_classes=11)
        print(f"✅ DeepLabV3创建成功 - 参数量: {sum(p.numel() for p in deeplab.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward():
    """测试模型前向传播"""
    print_section("3. 测试模型前向传播")
    
    try:
        from models import get_model
        
        # 创建dummy input
        batch_size = 1
        channels = 3
        height = 768
        width = 768
        x = torch.randn(batch_size, channels, height, width)
        
        # 测试UNet
        unet = get_model('unet', num_classes=11)
        unet.eval()
        with torch.no_grad():
            y_unet = unet(x)
        print(f"✅ UNet前向传播成功 - 输出形状: {y_unet.shape}")
        
        # 测试PSPNet
        pspnet = get_model('pspnet', num_classes=11)
        pspnet.eval()
        with torch.no_grad():
            y_pspnet = pspnet(x)
        print(f"✅ PSPNet前向传播成功 - 输出形状: {y_pspnet.shape}")
        
        # 测试DeepLabV3
        deeplab = get_model('deeplabv3', num_classes=11)
        deeplab.eval()
        with torch.no_grad():
            y_deeplab = deeplab(x)
        print(f"✅ DeepLabV3前向传播成功 - 输出形状: {y_deeplab.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions():
    """测试损失函数"""
    print_section("4. 测试损失函数")
    
    try:
        from modules.loss import FocalLoss, DiceLoss, TverskyLoss, get_class_weights
        
        # 创建dummy predictions和labels
        batch_size = 2
        num_classes = 11
        height = 64
        width = 64
        
        preds = torch.randn(batch_size, num_classes, height, width)
        labels = torch.randint(0, num_classes, (batch_size, height, width))
        
        # 测试FocalLoss
        focal = FocalLoss(gamma=2.0)
        loss_focal = focal(preds, labels)
        print(f"✅ FocalLoss计算成功 - Loss值: {loss_focal.item():.4f}")
        
        # 测试DiceLoss
        dice = DiceLoss()
        loss_dice = dice(preds, labels)
        print(f"✅ DiceLoss计算成功 - Loss值: {loss_dice.item():.4f}")
        
        # 测试TverskyLoss
        tversky = TverskyLoss(alpha=0.6, beta=0.4)
        loss_tversky = tversky(preds, labels)
        print(f"✅ TverskyLoss计算成功 - Loss值: {loss_tversky.item():.4f}")
        
        # 测试类别权重
        weights = get_class_weights()
        print(f"✅ 类别权重计算成功 - 权重形状: {weights.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """测试评估指标"""
    print_section("5. 测试评估指标")
    
    try:
        from modules.metrics import pixel_accuracy_filtered, compute_miou
        import numpy as np
        
        # 创建dummy predictions和labels
        height = 256
        width = 256
        preds = np.random.randint(0, 11, (height, width))
        labels = np.random.randint(0, 11, (height, width))
        
        # 测试Pixel Accuracy
        acc = pixel_accuracy_filtered(preds, labels, include_classes=[2, 3, 4, 5])
        print(f"✅ Pixel Accuracy计算成功 - 准确率: {acc:.4f}")
        
        # 测试mIoU
        miou = compute_miou(preds, labels, num_classes=11, include_classes=[2, 3, 4, 5])
        print(f"✅ mIoU计算成功 - mIoU值: {miou:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ 评估指标测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configs():
    """测试配置加载"""
    print_section("6. 测试配置加载")
    
    try:
        from configs.dataset_config import DATASET_CONFIG, DATALOADER_CONFIG, TARGET_CLASSES
        from configs.model_config import SUPPORTED_MODELS, MODEL_CONFIG
        from configs.training_config import TRAINING_CONFIG, LOSS_CONFIG
        
        print(f"✅ 数据集配置加载成功:")
        print(f"   - 训练图像目录: {DATASET_CONFIG['train_img_dir']}")
        print(f"   - Batch Size: {DATALOADER_CONFIG['batch_size']}")
        
        print(f"\n✅ 模型配置加载成功:")
        print(f"   - 支持的模型: {SUPPORTED_MODELS}")
        print(f"   - 类别数: {MODEL_CONFIG['num_classes']}")
        
        print(f"\n✅ 训练配置加载成功:")
        print(f"   - Epochs: {TRAINING_CONFIG['epochs']}")
        print(f"   - 学习率: {TRAINING_CONFIG['learning_rate']}")
        print(f"   - 目标类别: {TARGET_CLASSES}")
        
        return True
    except Exception as e:
        print(f"❌ 配置加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("  项目重构验证测试")
    print("="*60)
    print(f"\nPython版本: {sys.version.split()[0]}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 运行所有测试
    results = []
    results.append(("模块导入", test_imports()))
    results.append(("模型创建", test_model_creation()))
    results.append(("模型前向传播", test_model_forward()))
    results.append(("损失函数", test_loss_functions()))
    results.append(("评估指标", test_metrics()))
    results.append(("配置加载", test_configs()))
    
    # 打印总结
    print_section("测试结果总结")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20s} : {status}")
    
    print("\n" + "-"*60)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！项目重构成功！")
        print("\n下一步:")
        print("  1. 阅读 README.md 了解使用方法")
        print("  2. 查看 QUICKSTART.md 快速开始")
        print("  3. 运行 python train.py 开始训练")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查错误信息")
    
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
