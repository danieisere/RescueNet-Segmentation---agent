"""会话层的受限目标协议；模型只能声明意图，不能指定文件或工具。"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from modules.agent.task_schema import TaskSpec


class FactChange(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    operation: Literal['add', 'correct']
    key: str = Field(min_length=1)
    value: str = Field(min_length=1)
    unit: str = ''
    location: str = ''
    scope: Literal['image', 'session'] = 'image'
    supersedes_fact_id: str | None = None
    verification: Literal['user_reported', 'confirmed', 'unverified'] = 'user_reported'
    observed_at: str = ''

    @model_validator(mode='after')
    def check_identity(self):
        if not self.key.strip() or not self.value.strip():
            raise ValueError('事实键与内容不能为空白')
        if self.operation == 'add' and self.supersedes_fact_id is not None:
            raise ValueError('新增事实不能指定被替代事实')
        return self


class DialoguePlan(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    intent: Literal['ask_question', 'provide_information', 'correct_information',
                    'change_focus', 'request_summary', 'generate_report',
                    'reset_session', 'switch_image']
    changes: list[FactChange] = Field(default_factory=list)
    focus: list[str | int] | None = None
    needs_analysis: bool = False
    task: TaskSpec | None = None
    clarification_question: str = ''
    unsupported_reason: str = ''
    decision_summary: str = ''

    @model_validator(mode='after')
    def check_routing(self):
        # 与内层共享关注主题校验，避免会话层产生内层无法执行的类别。
        if self.focus is not None:
            TaskSpec(focus=self.focus)
        if self.needs_analysis and (self.task is None or self.task.status != 'ready'):
            raise ValueError('需要分析时必须提供 ready 状态的 TaskSpec')
        if self.task is not None and self.task.status == 'ready' and not self.needs_analysis:
            raise ValueError('可执行任务必须显式声明 needs_analysis')
        if self.intent == 'generate_report' and not (self.clarification_question or self.unsupported_reason):
            if not self.needs_analysis or self.task is None or self.task.target != 'report':
                raise ValueError('报告意图必须请求 report 任务')
        if self.intent in ('reset_session', 'switch_image') and (self.needs_analysis or self.changes):
            raise ValueError('重置或切图不能同时写事实或启动分析')
        return self


class DiscussedPlan(BaseModel):
    """历史建议与现场事实分开，来源消息可回查，不能自动升级为用户决定。"""
    model_config = ConfigDict(extra='forbid', strict=True)

    content: str = Field(min_length=1)
    source_message_ids: list[int] = Field(min_length=1)
    image_id: str | None = None


class HistorySummary(BaseModel):
    """摘要仅保留讨论线索；事实量值始终由 active_facts 提供。"""
    model_config = ConfigDict(extra='forbid', strict=True)

    goals: list[str]
    focus: list[str]
    decisions: list[str]
    open_questions: list[str]
    corrections: list[str]
    sources: list[str]
    report_requirements: list[str]
    fact_references: list[str]
    # 默认空列表兼容旧摘要；新协议明确要求保存相关方案及来源。
    plans: list[DiscussedPlan] = Field(default_factory=list)
