"""多图解释只综合已有真实证据；不修改像素统计或规则评分。"""
import json
import re
from pydantic import BaseModel, ConfigDict, Field


class Finding(BaseModel):
    model_config = ConfigDict(extra='forbid')
    content: str = Field(min_length=1)
    source_image_ids: list[str] = Field(min_length=1)


class Allocation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    resource_fact_id: str
    image_id: str
    amount: float = Field(ge=0, allow_inf_nan=False)


class Synthesis(BaseModel):
    model_config = ConfigDict(extra='forbid')
    findings: list[Finding]
    recommendations: list[Finding]
    uncertainties: list[str]
    allocations: list[Allocation] = Field(default_factory=list)


def synthesize(client, question, evidence, snapshot):
    """有限重试并校验图片引用、共享资源分配，防止生成无依据的整体判断。"""
    allowed = {entry['image_id'] for entry in evidence}
    facts = {fact['fact_id']: fact for fact in snapshot.get('active_facts', []) if fact['scope'] == 'session'}
    system = '''你是灾情多图综合研判助手。只根据提供的每图真实结果与有效现场事实输出JSON。
不得编造图像观察，不拼接地理范围、不累加不同图片像素比例、不跨图按原始分数排名。
共同人力车辆是总资源，不是每图各拥有同样资源。建议不是已执行决定，现场信息标注用户提供尚未核实。
每条结论和建议必须填写确切source_image_ids。若量化分配共享资源，必须在allocations填写对应resource_fact_id、image_id、amount，且总额不得超过事实。
JSON结构：''' + json.dumps(Synthesis.model_json_schema(), ensure_ascii=False)
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(
        {'question': question, 'evidence': evidence, 'snapshot': snapshot}, ensure_ascii=False)}]
    for attempt in range(2):
        try:
            parsed = Synthesis.model_validate_json(client.complete(messages, max_tokens=4096, temperature=0.2, timeout=180))
            for finding in parsed.findings + parsed.recommendations:
                if not set(finding.source_image_ids) <= allowed:
                    raise ValueError('综合结论引用未完成或不存在的图片')
            allocated = {}
            for allocation in parsed.allocations:
                if allocation.image_id not in allowed or allocation.resource_fact_id not in facts:
                    raise ValueError('资源分配来源不在已核对范围')
                key = allocation.resource_fact_id
                amount = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*', facts[key]['value'])
                if not amount:
                    raise ValueError('共享资源量无法确定，不得量化分配')
                allocated[key] = allocated.get(key, 0) + allocation.amount
                if allocated[key] > float(amount[1]):
                    raise ValueError('资源分配超过共享总量')
            return parsed.model_dump()
        except (ValueError, TypeError):
            if attempt:
                raise ValueError('多图综合研判两次校验未通过，分图结果已保留') from None
            messages.append({'role': 'user', 'content': '请修正JSON结构、图片引用及共享资源数量；所有引用必须来自输入证据。'})


def synthesis_text(result, names):
    lines = ['## 多图综合研判']
    for key, title in [('findings', '综合发现'), ('recommendations', '综合建议')]:
        if result.get(key):
            lines += ['### ' + title]
            for finding in result[key]:
                sources = '、'.join(names[image_id] for image_id in finding['source_image_ids'])
                lines.append(f"- {finding['content']}（来源：{sources}）")
    if result.get('uncertainties'):
        lines += ['### 待核实事项'] + ['- ' + value for value in result['uncertainties']]
    return '\n\n'.join(lines)
