"""ReAct 工具适配：综合分析逻辑由独立服务提供。"""
from typing import Dict

from modules.agent.base_tool import BaseTool
from modules.agent.multimodal_analysis import MultimodalAnalysisService


class MultimodalAnalysisTool(BaseTool):
    """只适配工具协议，便于服务日后直接接入 LangGraph 节点。"""

    def __init__(self, service: MultimodalAnalysisService = None):
        self.service = service if service is not None else MultimodalAnalysisService()

    @property
    def name(self) -> str:
        return 'multimodal_analyze'

    @property
    def description(self) -> str:
        return '在统计和优先级完成后，结合原图、分割图、叠加图及真实工具数据生成综合结论和救援建议。参数由系统注入，Action Input 使用 {}。'

    def execute(self, user_query: str, segmentation_result: Dict, statistics_result: Dict, priority_result: Dict) -> Dict:
        """将运行时证据传入服务，不复制或重算上游统计。"""
        return self.service.analyze(user_query, segmentation_result, statistics_result, priority_result)
