"""模型配置参数"""

# 支持的模型列表
SUPPORTED_MODELS = ['unet', 'pspnet', 'deeplabv3']

# 模型默认配置
MODEL_CONFIG = {
    'num_classes': 11,
    'backbone': 'resnet18',
    'pretrained': True,
}

# UNet特定配置
UNET_CONFIG = {
    'pyramid_pool_sizes': (1, 2, 3, 6),
    'pyramid_out_channels': 512,
}

# PSPNet特定配置
PSPNET_CONFIG = {
    'pool_sizes': (1, 2, 3, 6),
    'reduction_channels': 512,
}

# DeepLabV3特定配置
DEEPLABV3_CONFIG = {
    'aspp_out_channels': 256,
    'atrous_rates': [6, 12, 18],
}
