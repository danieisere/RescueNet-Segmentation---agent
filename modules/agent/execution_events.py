"""对外只提供公开业务进度；不序列化状态、图片或模型内部推理。"""
from datetime import datetime, timezone
import re


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def safe_text(value, limit=800):
    # 服务异常可能包含 URL/密钥；不让这些内容进入事件文件。
    text = str(value)
    text = re.sub(r'data:image/[^\s"\']+', '[image omitted]', text)
    text = re.sub(r'(?i)(?:sk-[\w-]+|https?://\S+)', '[redacted]', text)
    return text[:limit]


def event(run_id, item_id, node, kind, status, summary='', **extra):
    allowed = {key: value for key, value in extra.items()
               if key in ('elapsed_seconds', 'attempt', 'artifact_refs')}
    return {'schema_version': 1, 'run_id': run_id, 'item_id': item_id,
            'timestamp': utc_now(), 'node': node, 'event': kind, 'status': status,
            'summary': safe_text(summary), **allowed}
