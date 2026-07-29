"""训练配置参数"""

# 训练超参数
TRAINING_CONFIG = {
    'epochs': 50,
    'learning_rate': 5e-5,
    'weight_decay': 1e-4,
    'target_classes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 关注的类别
}

# 损失函数配置
LOSS_CONFIG = {
    'focal_gamma': 2.0,
    'tversky_alpha': 0.6,
    'tversky_beta': 0.4,
    'loss_weights': {
        'focal': 0.5,
        'tversky': 0.5,
    }
}

# 优化器配置
OPTIMIZER_CONFIG = {
    'optimizer_type': 'Adam',
    'lr_scheduler': 'ReduceLROnPlateau',
    'scheduler_patience': 3,
    'scheduler_factor': 0.5,
}

# 保存路径配置
SAVE_CONFIG = {
    'model_save_dir': './runs',
    'output_dir': './output',
    'save_interval': 10,  # 每隔多少个epoch保存一次
}
