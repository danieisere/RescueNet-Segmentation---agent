"""文件夹级运行档案：原子提交、相对引用、内容哈希及独立阶段产物。"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from .execution_events import utc_now, safe_text


def unique_id(prefix=''):
    local = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    return prefix + local + '_' + uuid4().hex[:10]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    # 只允许标量转换；意外传入数组必须报错，不能写入巨大的状态快照。
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError('档案不支持的对象类型：' + type(value).__name__)


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
    try:
        with open(temporary, 'xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, data):
    atomic_bytes(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False,
                                  default=json_default).encode('utf-8'))


class ArtifactStore:
    """一次 CLI 调用一个实例；每张图创建独立 ItemArtifacts。"""
    def __init__(self, output_root, engine, question, model_config=None):
        self.run_id = unique_id('run_')
        self.run_dir = Path(output_root).resolve() / self.run_id
        self.run_dir.mkdir(parents=True)
        self.engine, self.question = engine, question
        self.items = []
        self.sequence = 0
        # 白名单而非通用脱敏，避免未来配置新增凭据被意外保存。
        safe_config = {key: value for key, value in (model_config or {}).items()
                       if key in ('model', 'provider', 'reasoning_effort', 'api_mode')}
        self.manifest = {'schema_version': 1, 'run_id': self.run_id, 'engine': engine,
                         'question': question, 'created_at': utc_now(), 'status': 'running',
                         'model_config': safe_config, 'items': []}
        atomic_json(self.run_dir / 'request.json', {'question': question, 'engine': engine})
        atomic_bytes(self.run_dir / 'events.jsonl', b'')
        self._save_manifest()

    def _save_manifest(self):
        self.manifest['updated_at'] = utc_now()
        self.manifest['items'] = [dict(item_id=item.item_id, path=item.item_dir.relative_to(self.run_dir).as_posix(),
                                       status=item.manifest['status']) for item in self.items]
        atomic_json(self.run_dir / 'manifest.json', self.manifest)

    def create_item(self, image_path, explicit_output=None, source_name=None):
        item = ItemArtifacts(self, image_path, explicit_output, source_name)
        self.items.append(item)
        self._save_manifest()
        return item

    def finish(self, results):
        statuses = [result['status'] for result in results]
        status = ('failed' if 'failed' in statuses else
                  'completed' if statuses and all(s == 'completed' for s in statuses) else
                  'needs_clarification' if 'needs_clarification' in statuses else 'unsupported')
        self.manifest['status'] = status
        self.manifest['counts'] = {name: statuses.count(name) for name in
                                  ('completed', 'failed', 'needs_clarification', 'unsupported')}
        result = {'status': status, 'counts': self.manifest['counts'], 'results': results}
        try:
            atomic_json(self.run_dir / 'result.json', result)
            self._save_manifest()
        except OSError as exc:
            # 多文件不能同时原子提交；任一终态写入失败时，尽力撤回运行级成功标记。
            # 已完成 item 的证据保留；永久磁盘故障时由退出码 1 明确告知无法完整归档。
            failure = {'type': type(exc).__name__, 'message': safe_text(exc), 'retryable': False}
            self.manifest.update(status='failed', failure_stage='run_finalization', error=failure)
            result.update(status='failed', failure_stage='run_finalization', error=failure)
            for operation in (lambda: atomic_json(self.run_dir / 'result.json', result), self._save_manifest):
                try:
                    operation()
                except OSError:
                    pass
            raise


class ItemArtifacts:
    def __init__(self, store, image_path, explicit_output, source_name=None):
        self.store = store
        self.item_id = unique_id('item_')
        self.item_dir = store.run_dir / 'items' / self.item_id
        self.item_dir.mkdir(parents=True)
        self.sequence = 0
        self.manifest = {'schema_version': 1, 'run_id': store.run_id, 'item_id': self.item_id,
                         'engine': store.engine, 'question': store.question, 'created_at': utc_now(),
                         'status': 'running', 'source': {'path': str(Path(image_path).resolve()),
                         'sha256': sha256(image_path) if Path(image_path).is_file() else None,
                         # 仅保存显示名，真实证据仍由 source.path 和 source.sha256 绑定。
                         'source_name': source_name or Path(image_path).name},
                         'task': None, 'artifacts': []}
        atomic_json(self.item_dir / 'request.json', {'question': store.question, 'image_path': image_path,
                                                   'explicit_output': explicit_output,
                                                   'source_name': source_name or Path(image_path).name})
        atomic_bytes(self.item_dir / 'events.jsonl', b'')
        self._save_manifest()

    def _path(self, relative):
        path = (self.item_dir / relative).resolve()
        if not path.is_relative_to(self.item_dir.resolve()) or path == self.item_dir.resolve():
            raise ValueError('产物路径超出当前 item')
        return path

    def _save_manifest(self):
        self.manifest['updated_at'] = utc_now()
        atomic_json(self.item_dir / 'manifest.json', self.manifest)

    def register(self, relative, role, dependencies=None):
        path = self._path(relative)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('不能登记未提交或空的产物')
        record = {'role': role, 'path': path.relative_to(self.item_dir).as_posix(),
                  'sha256': sha256(path), 'bytes': path.stat().st_size,
                  'dependencies': dependencies or {}}
        self.manifest['artifacts'] = [a for a in self.manifest['artifacts'] if a['path'] != record['path']] + [record]
        self._save_manifest()
        return record

    def write_json(self, relative, data, role, dependencies=None):
        atomic_json(self._path(relative), data)
        return self.register(relative, role, dependencies)

    def append_event(self, event):
        self.sequence += 1
        self.store.sequence += 1
        current = {**event, 'sequence': self.sequence, 'run_sequence': self.store.sequence}
        encoded = (json.dumps(current, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        # JSONL 逐事件刷新，长耗时节点开始时便可观察；业务文件仍用原子替换。
        for path in (self.item_dir / 'events.jsonl', self.store.run_dir / 'events.jsonl'):
            with open(path, 'ab') as stream:
                stream.write(encoded)
                stream.flush()
        return current

    def save_stage(self, stage, value, dependencies=None):
        if stage == 'segmentation':
            import numpy as np
            visual = value['visualizations']
            for role, suffix in (('original', 'jpg'), ('segmentation', 'png'), ('overlay', 'png')):
                data = base64.b64decode(visual[role].split(',', 1)[1], validate=True)
                relative = f'images/{role}.{suffix}'
                atomic_bytes(self._path(relative), data)
                self.register(relative, role, dependencies)
            buffer = io.BytesIO()
            np.save(buffer, value['best_mask'], allow_pickle=False)
            atomic_bytes(self._path('data/mask.npy'), buffer.getvalue())
            self.register('data/mask.npy', 'mask', dependencies)
            metadata = {key: value for key, value in visual.items() if key not in ('original', 'segmentation', 'overlay')}
            metadata.update(best_model=value['best_model'], scores=value['scores'])
            self.write_json('data/visualization.json', metadata, 'visualization', dependencies)
        elif stage == 'report':
            relative = Path(value['report_path']).resolve().relative_to(self.item_dir).as_posix()
            self.register(relative, 'report', dependencies)
        else:
            self.write_json('data/' + stage + '.json', value, stage, dependencies)
        return list(self.manifest['artifacts'])

    def invalidate(self, stages):
        # 上游重算后删除旧引用和旧文件，避免清单把已失效成果当作当前结果。
        roles = set(stages)
        if 'segmentation' in roles:
            roles |= {'original', 'overlay', 'mask', 'visualization'}
        kept = []
        for record in self.manifest['artifacts']:
            if record['role'] in roles:
                self._path(record['path']).unlink(missing_ok=True)
            else:
                kept.append(record)
        self.manifest['artifacts'] = kept
        self._save_manifest()

    def verify(self):
        return all(self._path(a['path']).is_file() and sha256(self._path(a['path'])) == a['sha256']
                   for a in self.manifest['artifacts'])

    def finish(self, result):
        self.manifest.update(status=result['status'], failure_stage=result.get('failure_stage'),
                             error=result.get('error'), task=result.get('task'))
        atomic_json(self.item_dir / 'result.json', result)
        self._save_manifest()
        self.store._save_manifest()
