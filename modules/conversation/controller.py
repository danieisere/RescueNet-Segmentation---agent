"""外层会话图负责记忆和轮次；内层图按目标执行真实分析。"""
import json
from pathlib import Path
import sqlite3
from contextlib import closing
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langgraph.checkpoint.sqlite import SqliteSaver

from modules.agent.artifact_store import atomic_json
from modules.agent.execution_events import safe_text, utc_now
from modules.agent.runtime_services import RuntimeServices
from modules.agent.task_schema import TaskSpec
from .analysis_bridge import AnalysisBridge
from .dialogue import ContextManager, DialogueService
from .schema import DialoguePlan
from .store import SessionStore


class ConversationState(TypedDict, total=False):
    session_id: str
    turn_id: str
    question: str
    context: dict
    plan: dict
    analysis_result: dict | None
    answer: str
    status: str
    error: dict | None
    result: dict | None
    next_session_id: str | None


class _LazyClient:
    """纯命令和本地读取无需提前创建网络客户端。"""
    def __init__(self, services):
        from configs.agent_config import LLM_CONFIG
        self.services = services
        self.config = {'model': LLM_CONFIG.get('model')}

    def complete(self, *args, **kwargs):
        return self.services.client.complete(*args, **kwargs)


class ConversationController:
    def __init__(self, store, services=None, dialogue=None, context_manager=None, config=None):
        from configs.agent_config import CONVERSATION_CONFIG
        self.store = store
        self.services = services or RuntimeServices()
        self.config = {**CONVERSATION_CONFIG, **(config or {})}
        client = _LazyClient(self.services)
        self.dialogue = dialogue or DialogueService(client, self.config)
        self.context_manager = context_manager or ContextManager(client, self.config)
        self.bridge = AnalysisBridge(self.services)
        self.last_result = None
        self.turn = None
        self._override = None
        self._attached = False

    def _emit(self, node, kind, status, summary='', **extra):
        progress = {'session_id': self.store.session_id, 'turn_id': self.turn['turn_id'],
                    'timestamp': utc_now(), 'node': node, 'event': kind,
                    'status': status, 'summary': safe_text(summary), **extra}
        get_stream_writer()(progress)

    def _guard(self, name, function):
        def node(state):
            self._emit(name, 'node_start', 'running', '开始执行')
            try:
                update = function(state)
                self._emit(name, 'node_end', update.get('status', 'completed'), '执行完成')
                return update
            except Exception as exc:
                message = safe_text(exc)
                self._emit(name, 'node_error', 'failed', message)
                return {'status': 'failed', 'error': {'stage': name, 'type': type(exc).__name__, 'message': message},
                        'answer': '本轮未完成：' + message}
        return node

    def _evidence(self):
        meta = self.store.meta()
        evidence = self.bridge.latest_evidence(self.store, meta['current_image_id'])
        if evidence and 'analysis' in evidence:
            # 旧建议不能成为新事实；现场状态变化后只保留不受影响的统计证据。
            current = self._snapshot()
            if evidence.get('conversation_fingerprint') != self._fact_fingerprint(current):
                evidence = {k: v for k, v in evidence.items() if k != 'analysis'}
        return evidence

    @staticmethod
    def _fact_fingerprint(snapshot):
        from .analysis_bridge import fingerprint
        # 执行状态版本可能因澄清更新而变化；证据是否过期只比较分析相关内容。
        return fingerprint({key: snapshot.get(key) for key in ('image_id', 'focus', 'active_facts', 'decisions')})

    def _snapshot(self):
        meta = self.store.meta()
        facts = self.store.facts(image_id=meta['current_image_id'])
        return {'session_id': self.store.session_id, 'image_id': meta['current_image_id'],
                'state_version': meta['version'], 'focus': meta['current_focus'],
                'active_facts': facts,
                'decisions': [fact for fact in facts if fact['key'] in {'decision', 'user_decision', '用户决定'}]}

    def _prepare(self, state):
        if self._override and self._override.intent in {'reset_session', 'switch_image'}:
            # 本地显式命令不依赖旧对话或模型；即使历史预算耗尽也必须能新建/切图。
            return {'context': {'session': self.store.meta()}}
        return {'context': self._prepare_context()}

    def _prepare_context(self, force=False):
        """压缩事件走外层 LangGraph 自定义流，终端和审计日志看到同一次真实操作。"""
        return self.context_manager.prepare(self.store, analysis=self._evidence(), force=force,
            on_event=lambda kind, summary, **details: self._emit('prepare_context', kind, 'running'
                if kind != 'compression_end' else 'completed', summary, **details))

    def _classify(self, state):
        from modules.agent.llm_client import LLMContextLengthError
        context = state['context']
        try:
            plan = self._override or self.dialogue.classify(state['question'], context)
        except LLMContextLengthError:
            # 只对明确的接口长度超限追加一次更强压缩，不能把鉴权或网络错误伪装成预算问题。
            context = self._prepare_context(force=True)
            plan = self.dialogue.classify(state['question'], context)
        if not isinstance(plan, DialoguePlan):
            plan = DialoguePlan.model_validate(plan)
        return {'plan': plan.model_dump(), 'context': context}

    def _apply(self, state):
        plan = DialoguePlan.model_validate(state['plan'])
        if plan.clarification_question or plan.unsupported_reason:
            return {}
        meta = self.store.meta()
        try:
            if plan.changes:
                self.store.apply_changes(state['turn_id'], [change.model_dump() for change in plan.changes], meta['current_image_id'])
            if plan.focus is not None:
                self.store.update_meta(current_focus=plan.focus)
        except ValueError as exc:
            return {'plan': {**state['plan'], 'changes': [], 'needs_analysis': False, 'task': None,
                             'clarification_question': str(exc)}}
        return {}

    @staticmethod
    def _route(state):
        if state.get('error'):
            return 'failed'
        plan = state['plan']
        if plan.get('clarification_question') or plan.get('unsupported_reason'):
            return 'feedback'
        if plan['intent'] in {'reset_session', 'switch_image'}:
            return 'switch'
        return 'analyze' if plan['needs_analysis'] else 'answer'

    def _feedback(self, state):
        plan = state['plan']
        question = plan.get('clarification_question')
        if question:
            self.store.update_meta(pending_clarification={'question': question, 'original_request': state['question']})
        return {'answer': question or plan['unsupported_reason'],
                'status': 'needs_clarification' if question else 'unsupported'}

    def _switch(self, state):
        if state['plan']['intent'] == 'reset_session':
            new_store = SessionStore.create(self.store.session_dir.parent)
            return {'next_session_id': new_store.session_id, 'answer': '已保留当前会话。新会话：' + new_store.session_id}
        if not self._attached:
            return {'answer': '请使用 /image 图片路径，或在会话调用中附带新图片。', 'status': 'needs_clarification'}
        self.store.update_meta(pending_clarification=None)
        return {'answer': '已切换当前图片；旧图片及其专属事实保留，后续按新图片分析。'}

    def _analyze(self, state):
        meta = self.store.meta()
        if not meta['current_image_path']:
            self.store.update_meta(pending_clarification={'question': '请先提供需要分析的图片。', 'original_request': state['question']})
            return {'status': 'needs_clarification', 'answer': '请先使用 /image 图片路径 提供需要分析的图片。'}
        task = TaskSpec.model_validate(state['plan']['task'])
        if not task.focus and meta['current_focus']:
            task = task.model_copy(update={'focus': meta['current_focus']})
        if task.target == 'report' and not task.report_format:
            task = task.model_copy(update={'report_format': 'html'})
        pending = meta.get('pending_clarification')
        question = state['question']
        if pending and pending.get('original_request'):
            question = pending['original_request'] + '\n用户本轮补充：' + question
        # 报告承接本轮引用的历史方案；原始全文仍留在消息库，不把整段聊天塞入视觉请求。
        snapshot = self._snapshot()
        if task.target == 'report':
            snapshot['related_plan'] = self.context_manager.report_plan(self.store, state['context'])
            # 意图节点已经综合历史偏好和本轮要求；不把过去“先不生成”的阶段指令再交给报告正文。
            snapshot['report_requirements'] = {'current_request': question, 'format': task.report_format,
                                               'scope': task.decision_summary}
        for progress in self.bridge.run_stream(self.store, self.turn, meta['current_image_id'], meta['current_image_path'],
                                               task, question, snapshot):
            self._emit(progress['node'], progress['event'], progress['status'], progress.get('summary', ''),
                       scope='analysis', run_id=progress.get('run_id'), item_id=progress.get('item_id'))
        result = self.bridge.last_result
        if result['status'] != 'completed':
            return {'status': result['status'], 'analysis_result': result, 'answer': result['final_answer'],
                    'error': {'stage': 'analysis', 'message': result.get('error')}}
        self.store.update_meta(pending_clarification=None)
        return {'analysis_result': result, 'answer': result['final_answer']}

    def _answer(self, state):
        from modules.agent.llm_client import LLMContextLengthError
        context = self._prepare_context()
        try:
            answer = self.dialogue.answer(state['question'], context)
        except LLMContextLengthError:
            context = self._prepare_context(force=True)
            answer = self.dialogue.answer(state['question'], context)
        self.store.update_meta(pending_clarification=None)
        return {'context': context, 'answer': answer}

    def _finalize(self, state):
        result = {key: state.get(key) for key in ('session_id', 'turn_id', 'status', 'answer', 'error', 'analysis_result', 'next_session_id')}
        result['session_dir'] = str(self.store.session_dir)
        result['turn_dir'] = self.turn['turn_dir']
        try:
            self.store.finish_turn(state['turn_id'], state['answer'], state['status'], result)
        except OSError as exc:
            # 回答可能已提交 SQLite。先重建导出，再读取权威记录，不能伪造另一份终态。
            try:
                self.store.export()
            except OSError:
                try:
                    self.store.record_archive_failure(state['turn_id'], safe_text(exc))
                except OSError:
                    pass
            committed = self.store.get_turn(state['turn_id'])
            if 'result' not in committed:
                raise
            result = committed['result']
        return {'result': result, 'status': result['status'], 'error': result.get('error')}

    def _build(self, saver):
        graph = StateGraph(ConversationState)
        functions = {'prepare_context': self._prepare, 'classify': self._classify, 'apply_memory': self._apply,
                     'feedback': self._feedback, 'switch': self._switch, 'analyze': self._analyze,
                     'answer': self._answer, 'finalize': self._finalize, 'failed': self._finalize}
        for name, function in functions.items():
            graph.add_node(name, self._guard(name, function))
        graph.add_edge(START, 'prepare_context')
        for current, following in [('prepare_context', 'classify'), ('classify', 'apply_memory')]:
            graph.add_conditional_edges(current, lambda state, next_node=following: 'failed' if state.get('error') else next_node,
                                        ['failed', following])
        graph.add_conditional_edges('apply_memory', self._route, ['failed', 'feedback', 'switch', 'analyze', 'answer'])
        for name in ('feedback', 'switch', 'analyze', 'answer'):
            graph.add_conditional_edges(name, lambda state: 'failed' if state.get('error') else 'finalize', ['failed', 'finalize'])
        graph.add_edge('finalize', END)
        graph.add_edge('failed', END)
        return graph.compile(checkpointer=saver)

    def _append_event(self, progress):
        path = Path(self.turn['turn_dir']) / 'events.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        progress['sequence'] = self._sequence = self._sequence + 1
        with path.open('a', encoding='utf-8') as file:
            file.write(json.dumps(progress, ensure_ascii=False, allow_nan=False) + '\n')
        return progress

    def stream(self, question, image_path=None, request_id=None, plan_override=None):
        with self.store.lock():
            # 未取得锁的并发请求不得清空正在执行轮次的临时结果。
            self.last_result = None
            self.bridge.last_result = None
            self.store.mark_interrupted()
            override = plan_override
            if override is not None and not isinstance(override, DialoguePlan):
                override = DialoguePlan.model_validate(override)
            attachment = Path(image_path).expanduser().resolve() if image_path else None
            from modules.agent.artifact_store import sha256
            metadata = {'image_path': str(attachment) if attachment else None,
                        'image_sha256': sha256(attachment) if attachment and attachment.is_file() else None,
                        'local_plan': override.model_dump() if override else None}
            self.turn = self.store.begin_turn(question, request_id=request_id, input_metadata=metadata)
            self._sequence = 0
            if 'answer' in self.turn:
                # 已完成请求再次提交只读原结果，不能重新调用 LLM 或写入事实。
                self.last_result = self.turn['result']
                yield {'session_id': self.store.session_id, 'turn_id': self.turn['turn_id'], 'node': 'resume',
                       'event': 'request_reused', 'status': self.turn['status'], 'summary': '返回已归档的同一请求结果'}
                return
            self._override = override
            self._attached = False
            state = {'session_id': self.store.session_id, 'turn_id': self.turn['turn_id'], 'question': question,
                     'context': {}, 'plan': {}, 'analysis_result': None, 'answer': '', 'status': 'completed',
                     'error': None, 'result': None, 'next_session_id': None}
            try:
                if image_path:
                    self.store.attach_image(image_path)
                    self._attached = True
                # closing 负责释放句柄；不能在 sqlite 的事务上下文退出前先关闭连接。
                with closing(sqlite3.connect(self.store.session_dir / 'checkpoints.sqlite', check_same_thread=False)) as connection:
                    graph = self._build(SqliteSaver(connection))
                    for mode, data in graph.stream(state, stream_mode=['custom', 'updates'],
                                                  config={'configurable': {'thread_id': self.store.session_id}, 'recursion_limit': 40}):
                        if mode == 'custom':
                            yield self._append_event(data)
                        else:
                            for update in data.values():
                                if update and update.get('result'):
                                    self.last_result = update['result']
                if self.last_result is None:
                    raise RuntimeError('会话图结束但轮次未完成归档')
            except Exception as exc:
                committed = self.store.get_turn(self.turn['turn_id'])
                if 'answer' in committed:
                    # 末尾事件或检查点可能在回答提交后失败，先修复归档并读取权威结果。
                    try:
                        self.store.export()
                        progress = self._append_event({'session_id': self.store.session_id, 'turn_id': self.turn['turn_id'],
                            'node': 'recovery', 'event': 'runtime_recovered', 'status': committed['status'],
                            'timestamp': utc_now(), 'summary': '已从数据库恢复已提交结果：' + safe_text(exc)})
                        self.last_result = committed['result']
                        yield progress
                        return
                    except OSError:
                        try:
                            self.store.record_archive_failure(self.turn['turn_id'], safe_text(exc))
                        except OSError:
                            pass
                        self.last_result = self.store.get_turn(self.turn['turn_id'])['result']
                        yield {'session_id': self.store.session_id, 'turn_id': self.turn['turn_id'],
                               'node': 'archive', 'event': 'node_error', 'status': 'failed',
                               'summary': '结果已入库，但文件归档失败：' + safe_text(exc)}
                        return
                result = {'session_id': self.store.session_id, 'turn_id': self.turn['turn_id'], 'status': 'failed',
                          'answer': '本轮未完成：' + safe_text(exc), 'error': {'stage': 'conversation_runtime', 'type': type(exc).__name__, 'message': safe_text(exc)},
                          'analysis_result': self.bridge.last_result, 'session_dir': str(self.store.session_dir), 'turn_dir': self.turn['turn_dir']}
                self.last_result = result
                try:
                    self.store.finish_turn(self.turn['turn_id'], result['answer'], 'failed', result)
                    self.store.export()
                except Exception:
                    pass
                progress = {'session_id': self.store.session_id, 'turn_id': self.turn['turn_id'],
                            'node': 'conversation_runtime', 'event': 'node_error', 'status': 'failed',
                            'timestamp': utc_now(), 'summary': result['answer']}
                try:
                    progress = self._append_event(progress)
                except OSError:
                    # 无法写盘时仍把失败告知终端，不再让异常掩盖已保存的业务记录。
                    pass
                yield progress

    def run(self, question, **kwargs):
        for _ in self.stream(question, **kwargs):
            pass
        return self.last_result
