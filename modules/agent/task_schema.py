"""LLM 只描述目标；依赖闭包由程序计算，不接受任意工具或路径。"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

STAGES = ('segmentation', 'statistics', 'priority', 'analysis', 'report')


class TaskConflictError(ValueError):
    """合法字段之间存在语义冲突，有限修正后应询问用户而非当作执行失败。"""


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    target: Literal['segmentation', 'statistics', 'priority', 'report'] = 'segmentation'
    focus: list[str | int] = Field(default_factory=list)
    needs_visual_analysis: bool = False
    needs_rescue_advice: bool = False
    requested_artifacts: list[Literal['original', 'segmentation', 'overlay', 'mask', 'statistics', 'priority', 'analysis', 'report']] = Field(default_factory=list)
    report_format: Literal['html', 'markdown'] | None = None
    status: Literal['ready', 'needs_clarification', 'unsupported'] = 'ready'
    clarification_question: str = ''
    decision_summary: str = ''

    @model_validator(mode='after')
    def check_consistency(self):
        # 关注重点只允许已知类别或主题，避免把任意指令当作节点参数。
        topics = {'all', 'buildings', 'roads', 'water', 'vehicles', 'damage', 'rescue', 'vegetation', 'pool', 'background'}
        for value in self.focus:
            if isinstance(value, int):
                if not 0 <= value <= 10:
                    raise ValueError('类别编号必须为 0 至 10')
            elif value not in topics:
                raise ValueError('focus 必须使用约定主题或类别编号')
        if self.status != 'ready':
            if self.status == 'needs_clarification' and not self.clarification_question.strip():
                raise ValueError('澄清状态必须提出具体问题')
            return self
        if self.target == 'segmentation' and (self.needs_visual_analysis or self.needs_rescue_advice):
            raise TaskConflictError('分割目标不能同时要求统计解释或救援建议')
        if self.needs_rescue_advice and self.target not in ('priority', 'report'):
            raise TaskConflictError('救援建议目标必须包含优先级')
        if self.target != 'report' and (self.report_format or 'report' in self.requested_artifacts):
            raise TaskConflictError('报告产物必须对应 report 目标')
        allowed = set(required_stages(self)) | {'original', 'overlay', 'mask', 'segmentation'}
        if not set(self.requested_artifacts) <= allowed:
            raise TaskConflictError('请求产物与目标深度不一致')
        return self


def required_stages(task: TaskSpec) -> list[str]:
    """稳定顺序的依赖闭包，不因 LLM 漏报前置步骤而跳过真实数据计算。"""
    if task.status != 'ready':
        return []
    depth = {'segmentation': 0, 'statistics': 1, 'priority': 2, 'report': 4}[task.target]
    result = ['segmentation']
    if depth >= 1 or task.needs_visual_analysis:
        result.append('statistics')
    if depth >= 2 or task.needs_rescue_advice:
        result.append('priority')
    if task.needs_visual_analysis or task.needs_rescue_advice or task.target == 'report':
        result.append('analysis')
    if task.target == 'report':
        result.append('report')
    return result
