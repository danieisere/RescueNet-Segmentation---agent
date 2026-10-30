"""真正的多节点 StateGraph；消费原生 custom/updates 流并只导出安全事件。"""
from langgraph.graph import StateGraph, START, END
from configs.agent_config import GRAPH_CONFIG
from .graph_state import GraphState
from .graph_nodes import GraphNodes, route
from .task_schema import STAGES
from .execution_events import event, safe_text


class LangGraphController:
    def __init__(self, services, item):
        self.item, self.services = item, services
        self.last_result = None
        nodes = GraphNodes(services, item)
        self.nodes = nodes
        builder = StateGraph(GraphState)
        builder.add_node('validate', nodes.guarded('validate', nodes.validate))
        builder.add_node('classify', nodes.guarded('classify', nodes.classify))
        for stage in STAGES:
            builder.add_node(stage, nodes.guarded(stage, nodes.stage(stage)))
        builder.add_node('save_artifacts', nodes.guarded('save_artifacts', nodes.save_artifacts))
        for name, status in (('clarify', 'needs_clarification'), ('unsupported', 'unsupported'),
                             ('failed', 'failed'), ('finalize', None)):
            builder.add_node(name, nodes.guarded(name, lambda state, terminal=status: nodes.finalize(state, terminal)))
        builder.add_edge(START, 'validate')
        builder.add_conditional_edges('validate', lambda state: 'failed' if state.get('error') else 'classify', ['failed', 'classify'])
        destinations = [*STAGES, 'save_artifacts', 'finalize', 'failed', 'clarify', 'unsupported']
        for name in ('classify', *STAGES, 'save_artifacts'):
            builder.add_conditional_edges(name, route, destinations)
        for name in ('clarify', 'unsupported', 'failed'):
            builder.add_edge(name, END)
        builder.add_conditional_edges('finalize', lambda state: 'failed' if state.get('error') else END, ['failed', END])
        self.graph = builder.compile()

    def stream(self, question, context):
        state = {'question': question, 'image_path': context['image_path'], 'run_id': self.item.store.run_id,
                 'item_id': self.item.item_id, 'explicit_output': context.get('explicit_output'),
                 'output_filename': context.get('output_filename'), 'pending': [], 'events': [],
                 'versions': {}, 'signatures': {}, 'stage_status': {}, 'status': 'running'}
        shadow = dict(state)
        try:
            for mode, data in self.graph.stream(state, stream_mode=['custom', 'updates'],
                                                config={'recursion_limit': int(GRAPH_CONFIG['max_steps'])}):
                if mode == 'custom':
                    yield self.item.append_event(data)
                else:
                    # updates 含真实数组和 Base64，只合并到内存，公开部分只保留节点及字段名。
                    for node, update in data.items():
                        if update:
                            shadow.update(update)
                            summary = event(state['run_id'], state['item_id'], node, 'state_update',
                                            update.get('status', 'running'), '状态字段：' + ', '.join(k for k in update if k != 'events'))
                            yield self.item.append_event(summary)
                            if update.get('result'):
                                self.last_result = update['result']
            if self.last_result is None:
                raise RuntimeError('图已结束但未成功归档最终状态')
        except Exception as exc:
            # 图级异常（含保存失败/步数上限）不能被 END 掩盖；尽力保存可用档案。
            message = safe_text(exc)
            result = {'status': 'failed', 'engine': 'langgraph', 'run_id': state['run_id'], 'item_id': state['item_id'],
                      'failure_stage': shadow.get('failure_stage', 'graph_runtime'),
                      'error': {'type': type(exc).__name__, 'message': message, 'retryable': False},
                      'task': shadow.get('task'), 'required_stages': shadow.get('required_stages', []),
                      'stages': shadow.get('stage_status', {}), 'artifacts': list(self.item.manifest['artifacts']),
                      'item_dir': str(self.item.item_dir), 'final_answer': '任务失败：' + message, 'result_data': None}
            self.last_result = result
            try:
                self.item.finish(result)
                yield self.item.append_event(event(state['run_id'], state['item_id'], 'graph_runtime', 'node_error', 'failed', message))
            except OSError:
                pass

    def run(self, question, context):
        for _ in self.stream(question, context):
            pass
        return self.last_result
