"""跨图片复用的运行时服务；图状态不持有模型、设备或网络客户端。"""
from pathlib import Path

from .intent_router import IntentRouter


class RuntimeServices:
    def __init__(self, client=None, segmenter=None, analysis_service=None):
        self._client = client
        self._segmenter = segmenter
        self._analysis = analysis_service
        self._statistics = self._priority = self._report = None

    @property
    def client(self):
        if self._client is None:
            from .llm_client import LLMClient
            self._client = LLMClient()
        return self._client

    def classify(self, question, explicit_output=None, on_retry=None):
        return IntentRouter(self.client).classify(question, explicit_output, on_retry)

    def segment(self, image_path, source_name=None):
        if self._segmenter is None:
            from .tools.ensemble_segment_tool import EnsembleSegmentationTool
            self._segmenter = EnsembleSegmentationTool()
        result = dict(self._segmenter.execute(image_path=image_path))
        # logits 仅供分割模型选择；不让其进入图状态、事件或持久化。
        result.pop('all_predictions', None)
        # source_name 是用户可见的原始文件名；缺失时回退到当前输入路径文件名。
        result['source_name'] = source_name or Path(image_path).name
        return result

    def statistics(self, segmentation):
        if self._statistics is None:
            from .tools.statistics_tool import StatisticsAnalyzerTool
            self._statistics = StatisticsAnalyzerTool()
        return self._statistics.execute(seg_results=[segmentation])

    def priority(self, statistics):
        if self._priority is None:
            from .tools.priority_tool import PriorityCalculatorTool
            self._priority = PriorityCalculatorTool()
        return self._priority.execute(stats_result=statistics)

    def analyze(self, **kwargs):
        if self._analysis is None:
            from .multimodal_analysis import MultimodalAnalysisService
            self._analysis = MultimodalAnalysisService(client=self.client)
        return self._analysis.analyze(**kwargs)

    def report(self, **kwargs):
        if self._report is None:
            from .tools.report_tool import ReportGeneratorTool
            self._report = ReportGeneratorTool()
        return self._report.execute(**kwargs)

    def react_tools(self):
        """旧引擎也按需加载工具，批量图片复用同一组服务。"""
        from .base_tool import BaseTool

        class Adapter(BaseTool):
            def __init__(self, name, description, function):
                self._name, self._description, self.function = name, description, function

            @property
            def name(self):
                return self._name

            @property
            def description(self):
                return self._description

            def execute(self, **kwargs):
                return self.function(**kwargs)

        return {tool.name: tool for tool in (
            Adapter('ensemble_segment', '分割当前单图，生成原图、分割图、叠加图与真实掩膜。', lambda image_path: self.segment(image_path)),
            Adapter('analyze_statistics', '统计真实类别数组的像素数量与比例。', lambda seg_results: self.statistics(seg_results[0])),
            Adapter('calculate_priority', '基于统计计算规则优先级。', lambda stats_result: self.priority(stats_result)),
            Adapter('multimodal_analyze', '结合三图、统计和优先级生成完整救援综合分析。', self.analyze),
            Adapter('generate_report', '复用综合分析正文生成当前单图的独立应急报告。', self.report),
        )}
