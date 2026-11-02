"""FastAPI入口只适配HTTP，不在路由中复制Agent执行逻辑。"""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from modules.conversation.store import SessionStore
from configs.config_loader import CONVERSATION_CONFIG, load_config, _find_config, PROJECT_ROOT
from .schemas import MessageRequest, SettingsPatch
from .jobs import JobManager, JobConflict
from .uploads import receive_uploads, UploadTooLarge
from .resources import resource_index, public_resource, resolve_resource
from .events import event_stream


def session_summary(store):
    meta = store.meta()
    created = datetime.fromisoformat(meta['created_at']).astimezone(timezone(timedelta(hours=8)))
    jobs = store.web_rows('web_jobs')
    job = max(jobs, key=lambda job: job['created_at']) if jobs else None
    return dict(session_id=store.session_id, created_at=meta['created_at'], display_name=created.strftime('%Y-%m-%d %H:%M:%S'),
                last_turn=meta['last_turn'], image_count=len(store.images()), active_image_ids=meta['active_image_ids'],
                job_status=job['status'] if job else None, job_id=job['job_id'] if job else None)


def create_app(message_root=None, *, manager=None, config_path=None):
    root = Path(message_root or CONVERSATION_CONFIG['message_root']).resolve()
    runtime_config = deepcopy(load_config())
    job_manager = manager or JobManager(root, max_images=runtime_config.get('vision', {}).get('max_input_images', 5))

    @asynccontextmanager
    async def lifespan(app):
        root.mkdir(parents=True, exist_ok=True)
        job_manager.recover()
        yield
        await asyncio.to_thread(job_manager.close)

    app = FastAPI(title='RescueNet 灾情分析', lifespan=lifespan)
    app.state.manager = job_manager
    app.state.message_root = root
    app.state.runtime_config = runtime_config
    app.state.settings = None

    def settings_service():
        if app.state.settings is None:
            from .settings import SettingsService
            app.state.settings = SettingsService(config_path or _find_config(), runtime_config=runtime_config)
        return app.state.settings

    # 启动时固定配置版本，磁盘后续变更不能被误认成已生效。
    settings_service()

    def store_for(sid):
        try:
            return SessionStore(root, sid)
        except ValueError:
            raise FileNotFoundError('会话不存在') from None

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        code = 413 if isinstance(exc, UploadTooLarge) else 409 if isinstance(exc, JobConflict) else 400
        return JSONResponse({'error': {'code': type(exc).__name__, 'message': str(exc)}}, status_code=code)

    @app.exception_handler(FileNotFoundError)
    async def absent(request, exc):
        return JSONResponse({'error': {'code': 'not_found', 'message': str(exc)}}, status_code=404)

    @app.get('/api/meta')
    def meta():
        return dict(supported_formats=['.jpg', '.jpeg', '.png', '.bmp'],
                    max_images=runtime_config.get('vision', {}).get('max_input_images', 5),
                    max_upload_bytes=runtime_config.get('web', {}).get('max_upload_bytes', 20 * 1024 * 1024),
                    model=runtime_config.get('llm', {}).get('model', ''),
                    restart_required=settings_service().restart_required)

    @app.get('/api/sessions')
    def sessions(q: str = '', cursor: str = '', limit: int = Query(50, ge=1, le=200)):
        summaries = [session_summary(SessionStore(root, item['session_id'])) for item in SessionStore.list_sessions(root)]
        filtered = [value for value in summaries if (not cursor or value['session_id'] < cursor)
                    and (not q or q.lower() in (value['display_name'] + value['session_id']).lower())]
        page = filtered[:limit]
        return dict(sessions=page, next_cursor=page[-1]['session_id'] if len(filtered) > limit else None)

    @app.post('/api/sessions', status_code=201)
    def new_session():
        return session_summary(SessionStore.create(root))

    @app.get('/api/sessions/{sid}')
    def session(sid: str):
        store = store_for(sid)
        images = [{key: image.get(key) for key in ['image_id', 'source_name', 'relative_path', 'sha256']} for image in store.images()]
        return {**session_summary(store), 'images': images}

    @app.get('/api/sessions/{sid}/messages')
    def messages(sid: str, after_message_id: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500)):
        store = store_for(sid)
        scopes = store.message_image_scopes()
        values = [value for value in store.messages() if value['id'] > after_message_id]
        page = values[:limit]
        return dict(messages=[{**value, 'image_ids': scopes.get(value['id'], [])} for value in page],
                    next_after=page[-1]['id'] if len(values) > limit else None)

    @app.post('/api/sessions/{sid}/uploads', status_code=201)
    async def uploads(sid: str, files: list[UploadFile] = File(...), relative_paths: str = Form('[]')):
        try:
            paths = json.loads(relative_paths)
            if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
                raise ValueError('相对显示路径必须是字符串数组')
            result = await receive_uploads(store_for(sid), files, paths,
                runtime_config.get('web', {}).get('max_upload_bytes', 20 * 1024 * 1024),
                runtime_config.get('vision', {}).get('max_input_images', 5))
            return {'uploads': result}
        finally:
            for file in files:
                await file.close()

    @app.post('/api/sessions/{sid}/messages', status_code=202)
    def submit(sid: str, message: MessageRequest):
        return job_manager.enqueue(sid, message)

    @app.get('/api/sessions/{sid}/jobs/{jid}')
    def job(sid: str, jid: str):
        return job_manager.snapshot(sid, jid)

    @app.post('/api/sessions/{sid}/jobs/{jid}/retry', status_code=202)
    def retry_job(sid: str, jid: str):
        return job_manager.retry(sid, jid)

    @app.get('/api/sessions/{sid}/jobs/{jid}/events')
    async def events(sid: str, jid: str, request: Request, after: int = Query(0, ge=0)):
        job_manager.snapshot(sid, jid)
        cursor = request.headers.get('last-event-id')
        if cursor:
            try:
                after = max(after, int(cursor))
            except ValueError:
                raise ValueError('事件游标必须为非负整数') from None
        return StreamingResponse(event_stream(request, job_manager, store_for(sid), jid, after),
                                 media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.get('/api/sessions/{sid}/resources')
    def resources(sid: str, turn_id: str = '', image_id: str = '', group_id: str = '', kind: str = ''):
        values = resource_index(store_for(sid))
        return {'resources': [public_resource(value) for value in values
                if (not turn_id or value['turn_id'] == turn_id) and (not image_id or value['image_id'] == image_id)
                and (not group_id or group_id in value.get('group_ids', [])) and (not kind or value['kind'] == kind)]}

    @app.get('/api/sessions/{sid}/resources/{rid}')
    def resource(sid: str, rid: str, download: bool = False):
        value = resolve_resource(store_for(sid), rid)
        headers = {'X-Content-Type-Options': 'nosniff'}
        if value['_path'].suffix.lower() == '.html':
            headers['Content-Security-Policy'] = "sandbox; default-src 'none'; img-src data: 'self'; style-src 'unsafe-inline'; frame-src 'self'"
        return FileResponse(value['_path'], filename=value['_path'].name if download else None, headers=headers)

    @app.get('/api/settings')
    def settings():
        return settings_service().read()

    @app.patch('/api/settings')
    def save_settings(patch: SettingsPatch):
        return settings_service().save(patch.version, patch.changes)

    web = Path(PROJECT_ROOT) / 'web'
    if web.is_dir():
        app.mount('/static', StaticFiles(directory=web), name='static')

    @app.get('/')
    def index():
        return FileResponse(web / 'index.html')

    return app
