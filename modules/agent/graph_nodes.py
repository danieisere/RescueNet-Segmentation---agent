"""显式图节点和确定性路由：依据真实结果及版本选择执行，失败统一终止。"""
from pathlib import Path
from time import perf_counter
from langgraph.config import get_stream_writer
from .execution_events import event, safe_text
from .task_schema import STAGES, required_stages


def dependencies(stage, state):
    parents = {'segmentation': [], 'statistics': ['segmentation'], 'priority': ['statistics'],
               'analysis': ['segmentation', 'statistics'], 'report': ['segmentation', 'statistics', 'priority', 'analysis']}[stage]
    if stage == 'analysis' and ('priority' in state.get('required_stages', [])):
        parents = parents + ['priority']
    return {key: state.get('versions', {}).get(key, 0) for key in parents}


def is_current(stage, state):
    return (state.get(stage) is not None and state.get('stage_status', {}).get(stage, {}).get('status') == 'completed'
            and state.get('signatures', {}).get(stage) == dependencies(stage, state))


def route(state):
    """路由本身无模型调用，优先把新完成成果落盘，再进入下一个必要阶段。"""
    if state.get('error'):
        return 'failed'
    task = state.get('task', {})
    if task.get('status') == 'needs_clarification':
        return 'clarify'
    if task.get('status') == 'unsupported':
        return 'unsupported'
    if state.get('pending'):
        return 'save_artifacts'
    for stage in state.get('required_stages', []):
        if not is_current(stage, state):
            return stage
    return 'finalize'


def validate_result(stage, value, state):
    """完成不能只看字典非空；检查来源、掩膜统计及当前任务用途。"""
    import numpy as np
    image_path = Path(state['image_path']).resolve()
    if stage == 'segmentation':
        mask = value['best_mask']
        if (not isinstance(mask, np.ndarray) or mask.ndim != 2 or not mask.size
                or mask.dtype.kind not in 'iu' or mask.min() < 0 or mask.max() > 10):
            raise ValueError('分割结果不是有效类别掩膜')
        if Path(value['image_path']).resolve() != image_path or not value.get('visualizations'):
            raise ValueError('分割来源或三图可视化无效')
    elif stage == 'statistics':
        rows = value.get('per_image_stats', [])
        mask = state['segmentation']['best_mask']
        labels, counts = np.unique(mask, return_counts=True)
        expected = dict(zip(map(int, labels), map(int, counts)))
        if len(rows) != 1 or Path(rows[0]['image_path']).resolve() != image_path:
            raise ValueError('统计图像来源不一致')
        row = rows[0]
        if row['total_pixels'] != mask.size or {int(k): int(v) for k, v in row['class_counts'].items()} != expected:
            raise ValueError('统计与原始掩膜不一致')
        if any(not np.isclose(row['class_ratios'].get(k, row['class_ratios'].get(str(k), -1)), n / mask.size) for k, n in expected.items()):
            raise ValueError('统计占比与原始掩膜不一致')
    elif stage == 'priority':
        from configs.agent_config import PRIORITY_WEIGHTS as weights
        zones = value.get('priority_zones', [])
        if len(zones) != 1 or Path(zones[0]['image_path']).resolve() != image_path:
            raise ValueError('优先级来源不一致')
        counts = state['statistics']['per_image_stats'][0]['class_counts']
        score = sum(weights[key] * counts.get(label, 0) for key, label in
                    (('destroyed', 5), ('major', 4), ('minor', 3), ('road_blocked', 8)))
        level = 'CRITICAL' if score > 10000 else 'HIGH' if score > 5000 else 'MEDIUM' if score > 2000 else 'LOW'
        if not np.isclose(zones[0]['priority_score'], round(score, 2)) or zones[0]['level'] != level:
            raise ValueError('优先级与当前统计不一致')
    elif stage == 'analysis':
        mode = 'rescue' if state['task']['needs_rescue_advice'] or state['task']['target'] == 'report' else 'interpretation'
        if (value.get('status') != 'completed' or not value.get('conclusion') or value.get('mode') != mode
                or Path(value.get('image_path', '')).resolve() != image_path):
            raise ValueError('综合分析未完整完成或用途不符')
    elif stage == 'report':
        path = Path(value.get('report_path', ''))
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('报告文件未成功写入')


class GraphNodes:
    def __init__(self, services, item):
        self.services, self.item = services, item

    def emit(self, state, node, kind, status, summary='', **extra):
        current = event(state['run_id'], state['item_id'], node, kind, status, summary, **extra)
        get_stream_writer()(current)
        return current

    def guarded(self, name, function):
        def node(state):
            started = perf_counter()
            first = self.emit(state, name, 'node_start', 'running', '开始执行', attempt=1)
            try:
                update = function(state)
                status = update.get('status', 'completed') if name in ('finalize', 'failed', 'clarify', 'unsupported') else 'completed'
                last = self.emit(state, name, 'node_end', status, update.get('_summary', '执行完成'),
                                 elapsed_seconds=round(perf_counter() - started, 3), attempt=1)
                update.pop('_summary', None)
                update['events'] = [first, last]
                return update
            except Exception as exc:
                # 异常只返回可公开信息，所有下游业务边由错误状态阻断。
                message = safe_text(str(exc))
                last = self.emit(state, name, 'node_error', 'failed', f'{type(exc).__name__}: {message}',
                                 elapsed_seconds=round(perf_counter() - started, 3), attempt=1)
                stages = dict(state.get('stage_status', {}))
                stages[name] = {'status': 'failed', 'attempts': stages.get(name, {}).get('attempts', 0) + 1,
                                'elapsed_seconds': round(perf_counter() - started, 3)}
                return {'failure_stage': name, 'error': {'type': type(exc).__name__, 'message': message,
                         'retryable': name in ('classify', 'analysis')}, 'stage_status': stages,
                        'status': 'failed', 'events': [first, last]}
        return node

    def validate(self, state):
        from PIL import Image
        path = Path(state['image_path'])
        if not state['question'].strip():
            raise ValueError('用户问题不能为空')
        if not path.is_file():
            raise ValueError('图像文件不存在')
        with Image.open(path) as image:
            image.verify()
        source_hash = self.item.manifest['source']['sha256']
        from .artifact_store import sha256
        if source_hash != sha256(path):
            raise ValueError('输入图像在创建任务后发生变化')
        return {'status': 'running'}

    def classify(self, state):
        task = self.services.classify(state['question'], state.get('explicit_output'),
                                     lambda attempt: self.emit(state, 'classify', 'retry', 'running', '修正目标 JSON', attempt=attempt))
        return {'task': task.model_dump(), 'required_stages': required_stages(task), '_summary': task.decision_summary or task.status}

    def stage(self, name):
        def execute(state):
            signature = dependencies(name, state)
            for parent in signature:
                if not is_current(parent, state):
                    raise ValueError('缺少有效依赖：' + parent)
            started = perf_counter()
            if name == 'segmentation':
                value = self.services.segment(state['image_path'])
            elif name == 'statistics':
                value = self.services.statistics(state['segmentation'])
            elif name == 'priority':
                value = self.services.priority(state['statistics'])
            elif name == 'analysis':
                mode = 'rescue' if state['task']['needs_rescue_advice'] or state['task']['target'] == 'report' else 'interpretation'
                value = self.services.analyze(user_query=state['question'], segmentation_result=state['segmentation'],
                                              statistics_result=state['statistics'], priority_result=state.get('priority'), mode=mode,
                                              on_retry=lambda info: self.emit(state, 'analysis', 'retry', 'running',
                                                                              '修正综合分析 JSON', attempt=info['attempt']))
            else:
                value = self.services.report(priority_result=state['priority'], segmentation_result=state['segmentation'],
                                             statistics_result=state['statistics'], analysis_result=state['analysis'],
                                             output_filename=state.get('output_filename'), output_dir=str(self.item.item_dir / 'report'),
                                             report_format=state['task']['report_format'], save_analysis=False)
            validate_result(name, value, state)
            versions = dict(state.get('versions', {}))
            signatures = dict(state.get('signatures', {}))
            statuses = dict(state.get('stage_status', {}))
            update = {}
            downstream = list(STAGES[STAGES.index(name) + 1:])
            if any(state.get(key) is not None for key in downstream):
                self.item.invalidate(downstream)
            for key in downstream:
                update[key] = None
                versions.pop(key, None)
                signatures.pop(key, None)
                statuses.pop(key, None)
            versions[name] = versions.get(name, 0) + 1
            signatures[name] = signature
            statuses[name] = {'status': 'completed', 'attempts': state.get('stage_status', {}).get(name, {}).get('attempts', 0) + 1,
                              'elapsed_seconds': round(perf_counter() - started, 3)}
            return {**update, name: value, 'versions': versions, 'signatures': signatures, 'stage_status': statuses,
                    'pending': [name], '_summary': '当前版本完成；待独立归档'}
        return execute

    def save_artifacts(self, state):
        for stage in state.get('pending', []):
            if not is_current(stage, state):
                raise ValueError('拒绝保存过期结果：' + stage)
            refs = self.item.save_stage(stage, state[stage], state['signatures'][stage])
            self.emit(state, 'save_artifacts', 'artifact_saved', 'completed', stage,
                      artifact_refs=[r['path'] for r in refs])
        return {'pending': [], 'artifacts': list(self.item.manifest['artifacts'])}

    def finalize(self, state, terminal=None):
        status = terminal or 'completed'
        if status == 'completed':
            if state.get('pending') or not state.get('required_stages'):
                raise ValueError('任务尚有未保存或未识别的阶段')
            for stage in state['required_stages']:
                if not is_current(stage, state):
                    raise ValueError('目标阶段未完成：' + stage)
                validate_result(stage, state[stage], state)
            if not self.item.verify():
                raise ValueError('产物文件缺失或内容哈希不一致')
        task = state.get('task', {})
        if status == 'needs_clarification':
            answer = task.get('clarification_question', '请明确任务目标。')
        elif status == 'unsupported':
            answer = task.get('decision_summary') or '当前支持单图分割、像素统计、规则优先级、视觉解释和救援报告。'
        elif status == 'failed':
            answer = f"任务未完成。失败阶段：{state.get('failure_stage')}；{state.get('error', {}).get('message', '')}。已完成的产物已保留。"
        elif state.get('analysis'):
            answer = state['analysis']['conclusion']
        elif state.get('priority'):
            zone = state['priority']['priority_zones'][0]
            answer = f"规则优先级：{zone['level']}；评分：{zone['priority_score']}。该评分基于模型像素统计。"
        elif state.get('statistics'):
            from configs.dataset_config import CLASS_NAMES
            row = state['statistics']['per_image_stats'][0]
            lines = ['像素统计（不代表建筑栋数）：', '| 类别 | 像素 | 比例 |', '|---|---:|---:|']
            counts = row['class_counts']
            focus = task.get('focus', [])
            topics = {'buildings': [2, 3, 4, 5], 'roads': [7, 8], 'water': [1], 'vehicles': [6],
                      'damage': [3, 4, 5, 8], 'vegetation': [9], 'pool': [10], 'background': [0]}
            focused = set()
            for value in focus:
                focused.update([value] if isinstance(value, int) else topics.get(value, []))
            # 只筛选回答的展示重点，独立统计 JSON 与原始掩膜仍保留所有类别。
            labels = sorted(focused) if focused and not set(focus) & {'all', 'rescue'} else counts
            lines += [f"| {CLASS_NAMES[int(k)]} | {counts.get(k, 0)} | {counts.get(k, 0) / row['total_pixels']:.2%} |" for k in labels]
            answer = '\n'.join(lines)
        else:
            answer = '分割已完成，原图、类别图、半透明叠加图及原始掩膜已独立保存。'
        if state.get('report'):
            answer += '\n\n报告文件：' + state['report']['report_path']
        result = {'status': status, 'engine': self.item.store.engine, 'run_id': state['run_id'], 'item_id': state['item_id'],
                  'task': task, 'required_stages': state.get('required_stages', []), 'final_answer': answer,
                  'failure_stage': state.get('failure_stage'), 'error': state.get('error'),
                  'stages': state.get('stage_status', {}), 'versions': state.get('versions', {}),
                  'signatures': state.get('signatures', {}), 'artifacts': list(self.item.manifest['artifacts']),
                  'item_dir': str(self.item.item_dir), 'result_data': state.get('report')}
        self.item.finish(result)
        return {'result': result, 'status': status, 'final_answer': answer}
