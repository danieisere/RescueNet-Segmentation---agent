"""Agent模块初始化"""
from modules.agent.base_tool import BaseTool

__all__ = ['BaseTool', 'ReActController']


def __getattr__(name):
    # 保留旧导出，但仅在选择相应引擎时加载控制器及其依赖。
    if name == 'ReActController':
        from .react_controller import ReActController
        return ReActController
    if name == 'LangGraphController':
        from .graph_controller import LangGraphController
        return LangGraphController
    raise AttributeError(name)
