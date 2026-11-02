"""SSE仅观察SQLite事件，重连游标不触发重新分析。"""
import asyncio
import json


async def event_stream(request, manager, store, job_id, after):
    idle = 0
    while not await request.is_disconnected():
        events = store.web_events(job_id, after)
        for value in events:
            after = value['event_id']
            yield f'id: {after}\ndata: {json.dumps(value, ensure_ascii=False)}\n\n'
        job = manager.snapshot(store.session_id, job_id)
        if job['status'] not in {'queued', 'running'}:
            # 末尾事件可能刚刚与状态更新提交，补读一次避免提前结束连接。
            for value in store.web_events(job_id, after):
                yield f'id: {value["event_id"]}\ndata: {json.dumps(value, ensure_ascii=False)}\n\n'
            return
        idle += 1
        if idle % 30 == 0:
            yield ': heartbeat\n\n'
        await asyncio.sleep(0.3)
