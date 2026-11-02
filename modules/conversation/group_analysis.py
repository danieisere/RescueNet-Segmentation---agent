"""同一外层轮次按图调用既有分析桥接，分图计算与综合研判分开归档。"""
import json
from pathlib import Path
from modules.agent.artifact_store import atomic_json, sha256
from modules.agent.execution_events import safe_text
from .group_synthesis import synthesize, synthesis_text
from modules.agent.execution_events import utc_now
from .analysis_bridge import fingerprint


class GroupAnalysisService:
    def __init__(self, bridge, services):
        self.bridge, self.services = bridge, services
        self.last_result = None

    def run_stream(self, store, turn, image_ids, task, question, snapshot):
        group_id = 'group_' + turn['turn_id']
        directory = Path(turn['turn_dir']) / 'groups' / group_id
        directory.mkdir(parents=True, exist_ok=True)
        result = {'group_id': group_id, 'image_ids': image_ids, 'per_image': [], 'failures': [],
                  'aggregate_status': 'running', 'status': 'running', 'group_dir': str(directory),
                  'question': question, 'snapshot': snapshot, 'artifacts': []}
        result.update(created_at=utc_now(), turn_id=turn['turn_id'])
        peer_names = [snapshot.get('source_names', {}).get(image_id, store.image_metadata(image_id)['source_name']) for image_id in image_ids]
        atomic_json(directory / 'request.json', {'image_ids': image_ids, 'question': question, 'task': task.model_dump()})
        names, evidence = {}, []
        for index, image_id in enumerate(image_ids, 1):
            image = store.image_metadata(image_id)
            image['source_name'] = snapshot.get('source_names', {}).get(image_id, image['source_name'])
            names[image_id] = image['source_name']
            # 单图模型只接收当前图的影像；明确整轮其他图片由独立分图处理，避免把局部输入误写成全局证据缺失。
            image_question = (f"当前仅分析 {image['source_name']} 的影像、程序结果及本图有效现场事实。"
                              '其他图片由同轮独立分析，跨图比较和共享资源分配留给综合研判。'
                              '本次单图输入没有其他图片，不代表整轮缺少那些图片的证据。'
                              f'\n原始需求（只执行适用于本图的部分）：{question}')
            # 每图只得到本图事实与共同资源，不能携带另一图的专属事实。
            scoped = {**snapshot, 'image_id': image_id,
                      'source_name': image['source_name'],
                      'report_source_names': peer_names,
                      'active_facts': [fact for fact in snapshot['active_facts']
                                       if fact['scope'] == 'session' or fact.get('image_id') == image_id]}
            scoped['decisions'] = [fact for fact in scoped['active_facts'] if fact['key'] in {'decision', 'user_decision', '用户决定'}]
            if snapshot.get('report_requirements'):
                scoped['report_requirements'] = {**snapshot['report_requirements'], 'current_request': image_question,
                                                 'scope': f"仅生成 {image['source_name']} 的独立报告；跨图结论见综合报告"}
            yield {'node': 'group_analysis', 'event': 'image_start', 'status': 'running', 'image_id': image_id,
                   'group_id': group_id, 'summary': f"分析 {image['source_name']}（{index}/{len(image_ids)}）"}
            try:
                for progress in self.bridge.run_stream(store, turn, image_id, image['image_path'], task, image_question, scoped):
                    yield {**progress, 'image_id': image_id, 'group_id': group_id}
                item = dict(self.bridge.last_result)
            except Exception as exc:
                item = {'status': 'failed', 'final_answer': safe_text(exc), 'error': safe_text(exc), 'artifacts': []}
            item.update(image_id=image_id, source_name=image['source_name'])
            result['per_image'].append(item)
            if item['status'] == 'completed':
                entry = self.bridge.latest_evidence(store, image_id) or {}
                evidence.append({'image_id': image_id, 'source_name': image['source_name'], 'evidence': entry,
                                 'answer': item['final_answer'], 'signatures': item.get('signatures', {})})
            else:
                result['failures'].append({'image_id': image_id, 'message': item.get('final_answer', '分图分析未完成')})
        completed = len(evidence)
        result.update(status='completed' if completed == len(image_ids) else 'failed',
                      aggregate_status='completed' if completed == len(image_ids) else 'partial' if completed else 'failed')
        lines = [f"本轮 {len(image_ids)} 张航拍图，完成 {completed} 张。"]
        if completed:
            try:
                if task.needs_visual_analysis or task.needs_rescue_advice or task.target == 'report':
                    yield {'node': 'group_synthesis', 'event': 'node_start', 'status': 'running',
                           'group_id': group_id, 'summary': '正在依据各图结果综合研判'}
                    result['synthesis'] = synthesize(self.services.client, question, evidence, snapshot)
                    # 引用由真实清单补齐，模型只选择已完成图片，不能自行编造产物路径。
                    refs = {item['image_id']: [{'turn_id': turn['turn_id'], 'role': artifact['role'],
                        'sha256': artifact['sha256']} for artifact in item.get('artifacts', [])]
                        for item in result['per_image'] if item['status'] == 'completed'}
                    for finding in result['synthesis']['findings'] + result['synthesis']['recommendations']:
                        finding['evidence_refs'] = {image_id: refs[image_id] for image_id in finding['source_image_ids']}
                    lines.append(synthesis_text(result['synthesis'], names))
                else:
                    # 仅分割或规则统计任务采用真实结果汇编，不追加用户未要求的模型解释。
                    result['synthesis'] = {'findings': [], 'recommendations': [], 'uncertainties': [], 'allocations': []}
            except Exception as exc:
                result.update(status='failed', aggregate_status='partial', synthesis_error=safe_text(exc))
                lines.append('综合研判未完成：' + safe_text(exc) + '。已完成的分图成果已保留。')
        for item in result['per_image']:
            lines.append(f"### {item['source_name']}\n\n{item['final_answer']}")
        result['signature'] = fingerprint({'image_ids': image_ids, 'question': question, 'snapshot': snapshot,
            'evidence': [item.get('signatures', {}) for item in result['per_image']], 'task': task.model_dump(), 'renderer_version': 1})
        result['final_answer'] = '\n\n'.join(lines)
        atomic_json(directory / 'analysis.json', result)
        result['artifacts'].append({'role': 'analysis', 'path': 'analysis.json', 'sha256': sha256(directory/'analysis.json'),
                                    'bytes': (directory/'analysis.json').stat().st_size})
        if task.target == 'report' and completed and not result.get('synthesis_error'):
            try:
                from modules.agent.group_report import GroupReportRenderer
                report = GroupReportRenderer().render(result, directory / 'report', task.report_format or 'html')
                result['report'] = report
                result['artifacts'].extend(report['artifacts'])
            except Exception as exc:
                result.update(status='failed', aggregate_status='partial', report_error=safe_text(exc))
                result['final_answer'] += '\n\n综合报告未完成：' + safe_text(exc)
        atomic_json(directory / 'result.json', result)
        atomic_json(directory / 'manifest.json', {'status': result['status'], 'image_ids': image_ids,
                    'artifacts': result['artifacts'], 'per_image': [{'image_id': x['image_id'],
                    'item_dir': x.get('item_dir'), 'signatures': x.get('signatures', {})} for x in result['per_image']]})
        store.register_group(turn['turn_id'], result)
        self.last_result = result
