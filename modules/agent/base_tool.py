"""
Tool基类 - 所有工具必须继承此类

设计原则：
- 开闭原则：新增工具无需修改现有代码
- 单一职责：每个工具只做一件事
- 依赖倒置：高层模块不依赖低层模块细节
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseTool(ABC):
    """
    工具抽象基类
    
    所有Agent可调用的工具都必须继承此类，并实现：
    1. name属性：工具名称（用于LLM识别）
    2. description属性：工具描述（帮助LLM理解何时调用）
    3. execute方法：执行工具逻辑
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        工具名称（英文，无空格）
        
        示例：
        - 'ensemble_segment'
        - 'analyze_statistics'
        - 'calculate_priority'
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        工具描述（自然语言，帮助LLM理解功能）
        
        示例：
        - "并行调用UNet、PSPNet、DeepLabV3，选择置信度评分最高的模型结果"
        - "统计分割结果中各类别的数量和面积占比"
        """
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行工具
        
        Args:
            **kwargs: 工具所需的参数（由LLM提供）
        
        Returns:
            Dict: 工具执行结果（必须是字典格式，便于LLM解析）
        
        Raises:
            Exception: 工具执行失败时抛出异常
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
