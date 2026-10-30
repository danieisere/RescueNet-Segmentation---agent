"""业务状态保留真实数组；运行时模型与客户端由 RuntimeServices 持有。"""
from typing import Annotated, Any, TypedDict
from operator import add


class GraphState(TypedDict, total=False):
    question: str
    image_path: str
    run_id: str
    item_id: str
    explicit_output: str | None
    output_filename: str | None
    task: dict
    required_stages: list[str]
    segmentation: Any
    statistics: Any
    priority: Any
    analysis: Any
    report: Any
    versions: dict
    signatures: dict
    stage_status: dict
    pending: list[str]
    artifacts: list[dict]
    events: Annotated[list[dict], add]
    failure_stage: str
    error: dict
    status: str
    final_answer: str
    result: dict
