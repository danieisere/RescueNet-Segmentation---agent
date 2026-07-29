"""
数据集路径验证脚本
用于检查RescueNet数据集是否正确配置和可访问
"""
import os
from configs.dataset_config import DATASET_CONFIG, DATALOADER_CONFIG


def check_dataset_path(path_name, path_value):
    """检查单个数据集路径"""
    print(f"\n检查 {path_name}:")
    print(f"  路径: {path_value}")
    
    # 检查路径是否存在
    if not os.path.exists(path_value):
        print(f"  ❌ 路径不存在！")
        return False
    
    print(f"  ✅ 路径存在")
    
    # 统计文件数量
    files = [f for f in os.listdir(path_value) if f.endswith(('.jpg', '.png'))]
    print(f"  📁 文件数量: {len(files)}")
    
    if len(files) == 0:
        print(f"  ⚠️  警告: 目录为空！")
        return False
    
    # 显示前3个文件名
    print(f"  示例文件: {', '.join(files[:3])}")
    
    return True


def main():
    print("="*60)
    print("  RescueNet 数据集路径验证")
    print("="*60)
    
    print(f"\n📊 数据加载配置:")
    print(f"  Batch Size: {DATALOADER_CONFIG['batch_size']}")
    print(f"  Num Workers: {DATALOADER_CONFIG['num_workers']}")
    print(f"  Max Images: {DATALOADER_CONFIG['max_images']}")
    
    # 检查所有路径
    results = []
    for key, value in DATASET_CONFIG.items():
        result = check_dataset_path(key, value)
        results.append((key, result))
    
    # 打印总结
    print("\n" + "="*60)
    print("  验证结果总结")
    print("="*60)
    
    all_passed = all(result for _, result in results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20s} : {status}")
    
    print("\n" + "-"*60)
    if all_passed:
        print("🎉 所有数据集路径验证通过！可以开始训练/测试。")
        print("\n下一步:")
        print("  1. 运行 python train.py 开始训练")
        print("  2. 运行 python test.py 进行测试评估")
    else:
        print("⚠️  部分路径验证失败，请检查数据集是否完整。")
        print("\n提示:")
        print("  - 确认RescueNet文件夹位于项目根目录")
        print("  - 检查文件夹名称是否正确（区分大小写）")
        print("  - 确认图片文件格式为 .jpg 和 .png")
    
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
