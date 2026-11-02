"""网页任务的SQLite持久化扩展，复用SessionStore连接和事务边界。"""
import json
from contextlib import contextmanager


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class WebStoreMixin:
    @contextmanager
    def _web_db(self):
        with self._db() as connection:
            # 只在网页发生写操作时增量建表，读取历史会话不迁移或回写原文件。
            connection.execute('CREATE TABLE IF NOT EXISTS web_jobs (job_id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS web_uploads (upload_id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS web_events (job_id TEXT NOT NULL, sequence INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(job_id,sequence))')
            yield connection

    def web_rows(self, table):
        if table not in {'web_jobs', 'web_uploads'}:
            raise ValueError('未知网页索引')
        with self._db() as connection:
            if not connection.execute('SELECT 1 FROM sqlite_master WHERE type="table" AND name=?', (table,)).fetchone():
                return []
            return [json.loads(row[0]) for row in connection.execute(f'SELECT payload FROM {table}')]

    def save_upload(self, upload):
        with self._web_db() as connection:
            connection.execute('INSERT INTO web_uploads VALUES (?, ?)', (upload['upload_id'], encode(upload)))

    def create_job(self, job):
        with self._web_db() as connection:
            row = connection.execute('SELECT payload FROM web_jobs WHERE request_id=?', (job['request_id'],)).fetchone()
            if row:
                previous = json.loads(row[0])
                if previous['fingerprint'] != job['fingerprint']:
                    raise ValueError('request_id已用于不同的问题或图片集合')
                if previous['status'] != 'interrupted':
                    return previous, False
                # 仅显式重试中断任务；保持原job/turn身份，不能自动重跑已完成结果。
                job = {**previous, 'status': 'queued', 'error': None}
            for row in connection.execute('SELECT payload FROM web_jobs'):
                busy = json.loads(row[0])
                if busy['status'] in {'queued', 'running'}:
                    raise ValueError('当前会话仍有任务：' + busy['job_id'])
            connection.execute('INSERT OR REPLACE INTO web_jobs VALUES (?, ?, ?)', (job['job_id'], job['request_id'], encode(job)))
        return job, True

    def update_job(self, job_id, **fields):
        with self._web_db() as connection:
            row = connection.execute('SELECT payload FROM web_jobs WHERE job_id=?', (job_id,)).fetchone()
            if row is None:
                raise FileNotFoundError('任务不存在')
            job = json.loads(row[0])
            job.update(fields)
            connection.execute('UPDATE web_jobs SET payload=? WHERE job_id=?', (encode(job), job_id))
        return job

    def append_web_event(self, job_id, event):
        with self._web_db() as connection:
            number = connection.execute('SELECT COALESCE(MAX(sequence),0)+1 FROM web_events WHERE job_id=?', (job_id,)).fetchone()[0]
            value = {**event, 'event_id': number, 'job_id': job_id, 'session_id': self.session_id}
            connection.execute('INSERT INTO web_events VALUES (?, ?, ?)', (job_id, number, encode(value)))
        return value

    def web_events(self, job_id, after=0):
        with self._db() as connection:
            if not connection.execute('SELECT 1 FROM sqlite_master WHERE name="web_events"').fetchone():
                return []
            return [json.loads(row[0]) for row in connection.execute(
                'SELECT payload FROM web_events WHERE job_id=? AND sequence>? ORDER BY sequence', (job_id, after))]
