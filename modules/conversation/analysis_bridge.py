"""把可验证的会话成果恢复到内层分析图；新一轮独立归档，不改写历史。"""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from modules.agent.artifact_store import ArtifactStore, sha256
from modules.agent.graph_controller import LangGraphController
from modules.agent.graph_nodes import dependencies, validate_result, is_current
from modules.agent.task_schema import STAGES, TaskSpec, required_stages


def fingerprint(value):
    """稳定哈希只接收显式选择的配置和业务数据，不包含凭据。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_stage(item_dir, stage, manifest):
    """加载前验证当前阶段文件，禁止路径越界和 NumPy 对象反序列化。"""
    roles = {'segmentation': {'original', 'segmentation', 'overlay', 'mask', 'visualization'}}.get(stage, {stage})
    records = [record for record in manifest['artifacts'] if record['role'] in roles]
    if {record['role'] for record in records} != roles:
        raise ValueError('阶段档案不完整')
    paths = {}
    for record in records:
        path = (item_dir / record['path']).resolve()
        if not path.is_relative_to(item_dir) or not path.is_file() or sha256(path) != record['sha256']:
            raise ValueError('阶段文件来源或哈希无效')
        paths[record['role']] = path
    if stage == 'segmentation':
        import numpy as np
        metadata = json.loads(paths['visualization'].read_text(encoding='utf-8'))
        visual = {k: v for k, v in metadata.items() if k not in {'best_model', 'scores'}}
        for role, mime in [('original', 'jpeg'), ('segmentation', 'png'), ('overlay', 'png')]:
            visual[role] = f'data:image/{mime};base64,' + base64.b64encode(paths[role].read_bytes()).decode()
        return {'image_path': manifest['source']['path'], 'source_name': manifest['source'].get('source_name'),
                'best_model': metadata['best_model'],
                'scores': metadata['scores'], 'best_mask': np.load(paths['mask'], allow_pickle=False),
                'visualizations': visual}
    value = json.loads(paths[stage].read_text(encoding='utf-8'))
    if stage == 'statistics':
        # JSON 对象键变成字符串；恢复类别编号后再运行真实掩膜一致性校验。
        for row in value['per_image_stats']:
            for key in ('class_counts', 'class_ratios'):
                row[key] = {int(k): v for k, v in row[key].items()}
    return value


class AnalysisBridge:
    def __init__(self, services):
        self.services = services
        self.last_result = None
        self._weight_hashes = {}

    def contexts(self, image_path, task, question, conversation):
        from configs.agent_config import ENSEMBLE_CONFIG, PRIORITY_WEIGHTS, MULTIMODAL_CONFIG, LLM_CONFIG
        from configs.model_config import MODEL_CONFIG
        weights = []
        for name in ENSEMBLE_CONFIG['models']:
            path = Path(ENSEMBLE_CONFIG['weight_dir']) / f'{name}_final.pth'
            stamp = (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) if path.is_file() else (str(path), None)
            if stamp not in self._weight_hashes:
                self._weight_hashes[stamp] = sha256(path) if path.is_file() else None
            weights.append((name, self._weight_hashes[stamp]))
        # 预处理与可视化实现也进入签名，代码修改后不复用旧图或旧掩膜。
        agent_dir = Path(__file__).parents[1] / 'agent'
        source = [sha256(agent_dir / 'tools' / 'ensemble_segment_tool.py'), sha256(agent_dir / 'segmentation_visualization.py')]
        segmentation = fingerprint({'image': sha256(image_path), 'weights': weights, 'ensemble': ENSEMBLE_CONFIG,
                                    'models': MODEL_CONFIG, 'preprocessing': 'resize768-normalize-v1', 'source': source})
        return {'segmentation': segmentation,
                'statistics': fingerprint({'schema': 1, 'code': sha256(agent_dir / 'tools' / 'statistics_tool.py')}),
                'priority': fingerprint({'weights': PRIORITY_WEIGHTS, 'code': sha256(agent_dir / 'tools' / 'priority_tool.py')}),
                'analysis': fingerprint({'question': question, 'focus': task.focus, 'rescue': task.needs_rescue_advice or task.target == 'report',
                                         'conversation': conversation, 'model': LLM_CONFIG.get('model'),
                                         'multimodal': MULTIMODAL_CONFIG, 'code': sha256(agent_dir / 'multimodal_analysis.py')}),
                'report': fingerprint({'format': task.report_format, 'conversation': conversation})}

    def latest_evidence(self, store, image_id):
        """问答只读经过哈希检查的业务证据，不把数组、图片或旧事实送进上下文。"""
        for record in store.analyses(image_id):
            result = record['result']
            item_dir = Path(result['item_dir']).resolve()
            if not item_dir.is_relative_to(store.session_dir.resolve()):
                continue
            try:
                manifest = json.loads((item_dir / 'manifest.json').read_text(encoding='utf-8'))
                source = Path(manifest['source']['path']).resolve()
                if not source.is_relative_to(store.session_dir.resolve()) or sha256(source) != manifest['source']['sha256']:
                    continue
                snapshot = json.loads((item_dir / 'data' / 'conversation.json').read_text(encoding='utf-8'))
                task = TaskSpec.model_validate(result['task'])
                question = json.loads((item_dir / 'request.json').read_text(encoding='utf-8'))['question']
                state = {'image_path': str(source), 'task': task.model_dump(),
                         'required_stages': required_stages(task), 'versions': result.get('versions', {}),
                         'signatures': result.get('signatures', {}),
                         'dependency_context': self.contexts(str(source), task, question, snapshot)}
                evidence = {}
                for stage in state['required_stages']:
                    if stage == 'report' or result.get('stages', {}).get(stage, {}).get('status') != 'completed':
                        break
                    # 普通问答也检查当前配置及父签名，不能把旧规则评分当作当前证据。
                    if result.get('signatures', {}).get(stage) != dependencies(stage, state):
                        break
                    value = read_stage(item_dir, stage, manifest)
                    validate_result(stage, value, state)
                    state[stage] = value
                    if stage == 'segmentation':
                        continue
                    if stage == 'analysis':
                        value = {k: value[k] for k in ('conclusion', 'analysis', 'mode') if k in value}
                    evidence[stage] = value
                if evidence:
                    evidence['source_turn_id'] = record['turn_id']
                    evidence['conversation_fingerprint'] = fingerprint({key: snapshot.get(key) for key in ('image_id', 'focus', 'active_facts', 'decisions')})
                    return evidence
            except (OSError, ValueError, KeyError):
                continue
        return None

    def run_stream(self, store, turn, image_id, image_path, task, question, conversation_context):
        task = TaskSpec.model_validate(task.model_dump() if isinstance(task, TaskSpec) else task)
        self.last_result = None
        artifacts = ArtifactStore(Path(turn['turn_dir']) / 'artifacts', 'langgraph', question)
        image_metadata = store.image_metadata(image_id) or {}
        source_name = image_metadata.get('source_name')
        item = artifacts.create_item(image_path, source_name=source_name)
        contexts = self.contexts(image_path, task, question, conversation_context)
        state = {'image_path': image_path, 'source_name': source_name, 'task': task.model_dump(), 'required_stages': required_stages(task),
                 'dependency_context': contexts, 'versions': {}, 'signatures': {}, 'stage_status': {}}
        origins = {}
        # 每个阶段可从不同历史轮次读取；只复用当前需要的阶段，不携带旧报告。
        for stage in state['required_stages']:
            if stage == 'report':
                break
            for record in store.analyses(image_id):
                prior = record['result']
                if prior.get('stages', {}).get(stage, {}).get('status') != 'completed':
                    continue
                if prior.get('signatures', {}).get(stage) != dependencies(stage, state):
                    continue
                directory = Path(prior['item_dir']).resolve()
                if not directory.is_relative_to(store.session_dir.resolve()):
                    continue
                try:
                    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
                    value = read_stage(directory, stage, manifest)
                    validate_result(stage, value, state)
                except (OSError, ValueError, KeyError):
                    continue
                state[stage] = value
                state['versions'][stage] = prior['versions'][stage]
                state['signatures'][stage] = dependencies(stage, state)
                state['stage_status'][stage] = {'status': 'completed', 'reused': True, 'attempts': 0,
                                                'source_turn_id': record['turn_id']}
                item.save_stage(stage, value, state['signatures'][stage])
                origins[stage] = {'turn_id': record['turn_id'], 'item_dir': str(directory)}
                yield {'node': stage, 'event': 'artifact_reused', 'status': 'completed',
                       'summary': '已验证并复用历史成果：' + stage, 'source_turn_id': record['turn_id']}
                break
            if stage not in state:
                # 缺少上游就让真实图计算，不能跨过依赖恢复下游。
                break
        item.write_json('data/conversation.json', conversation_context, 'conversation')
        item.write_json('data/reuse.json', origins, 'reuse')
        controller = LangGraphController(self.services, item)
        context = {'image_path': image_path, 'task_override': task.model_dump(), 'restored_state': state,
                   'dependency_context': contexts, 'conversation_context': conversation_context}
        for progress in controller.stream(question, context):
            yield progress
        self.last_result = controller.last_result
        artifacts.finish([self.last_result])
        store.register_analysis(turn['turn_id'], image_id, self.last_result)
