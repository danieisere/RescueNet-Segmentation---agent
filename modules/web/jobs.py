"""提交和观察分离的后台队列；断开HTTP连接不会停止已提交分析。"""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import uuid

from modules.conversation.store import SessionStore
from modules.agent.execution_events import safe_text, utc_now
from modules.agent.artifact_store import sha256


class JobConflict(ValueError):
    """会话繁忙或同一幂等键被用于不同载荷。"""


class JobManager:
    def __init__(self, root, controller_factory=None, max_images=5):
        self.root = Path(root)
        self.controller_factory = controller_factory
        self.services = None
        self.max_images = max_images
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rescuenet-analysis')
        self._closed = False

    def _sources(self, store, images):
        uploads = {entry['upload_id']: entry for entry in store.web_rows('web_uploads')}
        sources = []
        for image in images:
            if image.kind == 'upload':
                if image.id not in uploads:
                    raise FileNotFoundError('上传附件不属于当前会话')
                value = uploads[image.id]
                value = self._archived_source(store, value)
                sources.append({key: value[key] for key in ['path', 'source_name', 'relative_path', 'sha256']})
            else:
                value = store.image_metadata(image.id)
                if not value:
                    raise FileNotFoundError('图片不属于当前会话')
                sources.append(dict(path=value['image_path'], source_name=value['source_name'],
                                    relative_path=value.get('relative_path', value['source_name']), sha256=value['sha256']))
        return sources

    @staticmethod
    def _archived_source(store, source):
        # 暂存清理后由内容哈希定位归档原图，原upload_id仍可幂等重放。
        if not Path(source['path']).is_file():
            image = next((image for image in store.images() if image['sha256'] == source['sha256']), None)
            if image is None:
                raise FileNotFoundError('附件暂存已失效且尚未归档，请重新选择图片')
            source = {**source, 'path': image['image_path']}
        return source

    @staticmethod
    def _cleanup_archived_uploads(store):
        archived = {image['sha256'] for image in store.images()}
        root = (store.session_dir / 'uploads').resolve()
        for upload in store.web_rows('web_uploads'):
            path = Path(upload['path']).resolve()
            # 只删除已有归档副本且内容校验一致的暂存文件，保留未归档失败输入供重试。
            if upload['sha256'] in archived and path.is_relative_to(root) and path.is_file() and sha256(path) == upload['sha256']:
                try:
                    path.unlink()
                    path.parent.rmdir()
                except OSError:
                    pass

    def enqueue(self, session_id, message_request):
        if self._closed:
            raise JobConflict('服务正在关闭，请稍后重试')
        store = SessionStore(self.root, session_id)
        sources = self._sources(store, message_request.images)
        if len(sources) > self.max_images:
            raise ValueError('超过单轮图片数量限制')
        # 指纹与暂存路径无关，相同字节重传可以返回原任务；顺序和原始名属于请求语义。
        previous = next((job for job in store.web_rows('web_jobs') if job['request_id'] == message_request.request_id), None)
        if previous is None or previous['status'] == 'interrupted':
            self._check_session_lock(store)
        payload = {'question': message_request.question,
                   'images': [{k: source[k] for k in ['sha256', 'source_name']} for source in sources],
                   'active_image_ids': [] if sources else previous['image_ids'] if previous else store.meta()['active_image_ids']}
        signature = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        job = dict(job_id='job_' + uuid.uuid4().hex, request_id=message_request.request_id,
                   fingerprint=signature, question=message_request.question, sources=sources,
                   image_ids=payload['active_image_ids'], status='queued', session_id=session_id,
                   created_at=utc_now(), turn_id=None)
        try:
            saved, created = store.create_job(job)
        except ValueError as exc:
            raise JobConflict(str(exc)) from None
        if created:
            self._event(store, saved, event='job_queued', node='queue', status='queued', summary='分析任务已排队')
            self.executor.submit(self._run, session_id, saved['job_id'])
        return self.snapshot(session_id, saved['job_id'])

    @staticmethod
    def _check_session_lock(store):
        try:
            with store.lock():
                pass
        except RuntimeError as exc:
            raise JobConflict(str(exc)) from None

    def snapshot(self, session_id, job_id):
        store = SessionStore(self.root, session_id)
        job = next((job for job in store.web_rows('web_jobs') if job['job_id'] == job_id), None)
        if job is None:
            raise FileNotFoundError('任务不存在')
        # 浏览器只获得结果和身份，不获得暂存文件的绝对路径。
        return {key: job.get(key) for key in ['job_id', 'session_id', 'turn_id', 'status', 'created_at', 'error', 'aggregate_status']}

    def retry(self, session_id, job_id):
        """浏览器只指定原任务，原问题、附件和幂等键从SQLite读取，不接受替换载荷。"""
        store = SessionStore(self.root, session_id)
        previous = next((job for job in store.web_rows('web_jobs') if job['job_id'] == job_id), None)
        if previous is None:
            raise FileNotFoundError('任务不存在')
        if self._closed:
            raise JobConflict('服务正在关闭')
        if previous['status'] == 'interrupted':
            self._check_session_lock(store)
        try:
            saved, created = store.create_job(previous)
        except ValueError as exc:
            raise JobConflict(str(exc)) from None
        if created:
            self._event(store, saved, event='job_queued', node='queue', status='queued', summary='正在以原载荷重试中断任务')
            self.executor.submit(self._run, session_id, saved['job_id'])
        return self.snapshot(session_id, job_id)

    def _event(self, store, job, **event):
        raw_sequence = event.pop('sequence', None)
        event.setdefault('timestamp', utc_now())
        event.setdefault('turn_id', job.get('turn_id'))
        if raw_sequence is not None:
            event['details'] = {'source_sequence': raw_sequence}
        value = store.append_web_event(job['job_id'], event)
        path = store.session_dir / 'web_jobs' / job['job_id'] / 'events.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        # SQLite是事件权威；导出失败不撤销已持久化事件，重连仍能读取。
        try:
            with path.open('a', encoding='utf-8') as output:
                output.write(json.dumps(value, ensure_ascii=False) + '\n')
        except OSError:
            pass
        return value

    def _run(self, session_id, job_id):
        store = SessionStore(self.root, session_id)
        job = store.update_job(job_id, status='running')
        try:
            if self.controller_factory is None:
                from modules.conversation.controller import ConversationController
                from modules.agent.runtime_services import RuntimeServices
                if self.services is None:
                    self.services = RuntimeServices()
                controller = ConversationController(store, services=self.services)
            else:
                controller = self.controller_factory(store)
            kwargs = {'request_id': job['request_id'], 'attachment_metadata': {'web_request': True,
                'sources': [{key: source[key] for key in ['source_name', 'relative_path', 'sha256']} for source in job['sources']]}}
            if job['sources']:
                kwargs['image_paths'] = [self._archived_source(store, source) for source in job['sources']]
            elif job['image_ids']:
                kwargs['image_ids'] = job['image_ids']
            for event in controller.stream(job['question'], **kwargs):
                if event.get('turn_id') and not job.get('turn_id'):
                    job = store.update_job(job_id, turn_id=event['turn_id'])
                self._event(store, job, **event)
            result = controller.last_result
            if result is None:
                raise RuntimeError('会话执行未返回已归档结果')
            group = result.get('analysis_result') or {}
            self._cleanup_archived_uploads(store)
            job = store.update_job(job_id, status=result['status'], turn_id=result['turn_id'],
                                   aggregate_status=group.get('aggregate_status'), error=result.get('error'))
            self._event(store, job, event='job_result', node='finalize', status=result['status'], summary='本轮已归档')
        except Exception as exc:
            job = store.update_job(job_id, status='failed', error=safe_text(exc))
            self._event(store, job, event='job_result', node='web_runtime', status='failed', summary=safe_text(exc))

    def recover(self):
        for meta in SessionStore.list_sessions(self.root):
            store = SessionStore(self.root, meta['session_id'])
            for job in store.web_rows('web_jobs'):
                if job['status'] == 'queued':
                    self.executor.submit(self._run, store.session_id, job['job_id'])
                elif job['status'] == 'running':
                    try:
                        with store.lock():
                            turn = store.get_turn(job['turn_id']) if job.get('turn_id') else {}
                            status = turn['status'] if 'answer' in turn else 'interrupted'
                            store.mark_interrupted()
                            result = turn.get('result') or {}
                            updated = store.update_job(job['job_id'], status=status, error=result.get('error'),
                                aggregate_status=(result.get('analysis_result') or {}).get('aggregate_status'))
                            self._event(store, updated, event='job_result', node='recovery', status=status,
                                        summary='已恢复归档结果' if status != 'interrupted' else '任务因服务重启中断，已保留落盘成果')
                    except RuntimeError:
                        # 锁仍由CLI或另一进程持有，不能把活跃任务标记为中断。
                        continue

    def close(self):
        self._closed = True
        self.executor.shutdown(wait=True)
