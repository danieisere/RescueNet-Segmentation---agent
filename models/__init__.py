from models.unet import UNet
from models.pspnet import PSPNet
from models.deeplabv3 import DeepLabV3_Custom


def get_model(model_name, num_classes=11):
    """统一模型加载接口"""
    if model_name == 'unet':
        return UNet(num_classes=num_classes)
    elif model_name == 'pspnet':
        return PSPNet(num_classes=num_classes, backbone='resnet18')
    elif model_name == 'deeplabv3':
        return DeepLabV3_Custom(num_classes=num_classes, backbone='resnet18')
    else:
        raise ValueError(f"Unknown model name {model_name}, expected 'unet', 'pspnet' or 'deeplabv3'.")
