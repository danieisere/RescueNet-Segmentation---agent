"""工具按需导入；校验/分类阶段不加载 torch 或模型。"""
from importlib import import_module
from modules.agent.base_tool import BaseTool

_MODULES = {
    'EnsembleSegmentationTool': 'ensemble_segment_tool',
    'StatisticsAnalyzerTool': 'statistics_tool',
    'PriorityCalculatorTool': 'priority_tool',
    'ReportGeneratorTool': 'report_tool',
    'MultimodalAnalysisTool': 'multimodal_tool',
}
__all__ = ['BaseTool', *_MODULES]


def __getattr__(name):
    if name in _MODULES:
        value = getattr(import_module('.' + _MODULES[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
