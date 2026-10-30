"""兼容训练模块导出；Agent 输入校验无需导入训练和绘图依赖。"""
from importlib import import_module

_EXPORTS = {
    'SegmentationDataset': 'datasets', 'FocalLoss': 'loss', 'DiceLoss': 'loss',
    'TverskyLoss': 'loss', 'get_class_weights': 'loss',
    'pixel_accuracy_filtered': 'metrics', 'compute_miou': 'metrics',
    'train': 'trainer', 'evaluate_model': 'evaluator', 'predict_and_draw': 'visualization',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module('.' + _EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
