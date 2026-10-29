"""分割工具和报告共用的图像编码、类别配色与空间对齐逻辑。"""
import base64
import io
from typing import Any, Dict

import numpy as np
from PIL import Image

from configs.dataset_config import CLASS_NAMES


# 沿用应急报告的 RGB 配色，模型输入、报告图像和图例只维护这一份映射。
SEGMENTATION_PALETTE = (
    (0, 0, 0), (40, 120, 220), (150, 150, 150), (255, 220, 0),
    (255, 120, 0), (220, 0, 0), (180, 0, 180), (40, 180, 80),
    (120, 70, 20), (0, 160, 80), (0, 220, 220),
)


def segmentation_legend() -> list:
    """提供完整的类别编号、名称和 RGB/十六进制颜色，便于模型与报告共同解释。"""
    return [
        {'class_id': class_id, 'name': CLASS_NAMES[class_id], 'rgb': list(color),
         'hex': '#{:02x}{:02x}{:02x}'.format(*color)}
        for class_id, color in enumerate(SEGMENTATION_PALETTE)
    ]


def colorize_mask(mask: np.ndarray) -> Image.Image:
    """将类别编号映射为 RGB；非法编号直接报错，避免悄悄映射成其他类别。"""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.size == 0 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError('分割掩膜必须是非空的二维整数数组')
    if mask.min() < 0 or mask.max() >= len(SEGMENTATION_PALETTE):
        raise ValueError('分割掩膜包含超出 RescueNet 类别范围的编号')
    return Image.fromarray(np.asarray(SEGMENTATION_PALETTE, dtype=np.uint8)[mask])


def image_to_data_url(image: Image.Image, image_format: str = 'PNG') -> str:
    """生成独立携带的图像内容，后续可直接用于多模态请求或嵌入报告。"""
    buffer = io.BytesIO()
    options = {'quality': 90} if image_format == 'JPEG' else {}
    image.save(buffer, format=image_format, optimize=True, **options)
    mime_type = 'jpeg' if image_format == 'JPEG' else 'png'
    return f'data:image/{mime_type};base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def build_segmentation_visualizations(
    original_rgb: np.ndarray,
    mask: np.ndarray,
    overlay_alpha: float = 0.5,
) -> Dict[str, Any]:
    """生成同一原图坐标系下的三图；所有 size/scale 均按宽、高排列。"""
    if not np.isfinite(overlay_alpha) or not 0 <= overlay_alpha <= 1:
        raise ValueError('叠加透明度必须在 0 到 1 之间')
    # 先验证原始编号，不能先转 uint8，以免越界值被截断后成为合法类别。
    colorize_mask(mask)
    original = Image.fromarray(original_rgb).convert('RGB')
    original_width, original_height = original.size
    mask_height, mask_width = mask.shape

    # 当前预处理是直接 resize 到 768×768，无裁剪/填充；分别按两个轴反向映射。
    # 仅缩放用于展示的副本，保留原 best_mask 的尺寸、内容和统计口径。
    display_mask = Image.fromarray(mask.astype(np.uint8)).resize(
        original.size, resample=Image.Resampling.NEAREST,
    )
    segmentation = colorize_mask(np.asarray(display_mask))
    overlay = Image.blend(original, segmentation, overlay_alpha)

    return {
        'original': image_to_data_url(original, 'JPEG'),
        'segmentation': image_to_data_url(segmentation),
        'overlay': image_to_data_url(overlay),
        'overlay_alpha': overlay_alpha,
        'legend': segmentation_legend(),
        'alignment': {
            'size_order': 'width,height',
            'original_size': [original_width, original_height],
            'mask_size': [mask_width, mask_height],
            'visualization_size': [original_width, original_height],
            'original_to_mask_scale': [mask_width / original_width, mask_height / original_height],
            'mask_to_original_scale': [original_width / mask_width, original_height / mask_height],
            'preprocessing': 'resize',
            'mask_resize': 'nearest',
        },
    }
