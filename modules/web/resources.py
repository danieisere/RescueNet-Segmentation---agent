"""以会话内清单构建资源索引，浏览器不能请求任意磁盘文件。"""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from modules.agent.artifact_store import sha256

LABELS = {'original': '原图', 'segmentation': '分割图', 'overlay': '叠加图', 'mask': '原始掩膜',
          'statistics': '像素统计', 'priority': '规则优先级', 'analysis': '综合分析',
          'visualization': '图例与空间映射', 'report': '灾情分析报告', 'group_report': '多图综合报告'}


def resource_index(store):
    root = store.session_dir.resolve()
    values = []
    groups = {}
    images = store.images()
    image_by_hash = {image['sha256']: image['image_id'] for image in images}

    def add(path, role, source_names, turn_id=None, image_id=None, group_id=None, digest=None, status='completed'):
        path = Path(path).resolve()
        if not path.is_relative_to(root) or not path.is_file() or role not in LABELS:
            return
        relative = path.relative_to(root).as_posix()
        rid = hashlib.sha256(relative.encode()).hexdigest()[:32]
        kind = 'reports' if role in {'report', 'group_report'} else 'images' if role in {'original', 'segmentation', 'overlay'} else 'data'
        name = path.name if kind == 'reports' else ('、'.join(source_names) + ' · ' + LABELS[role])
        values.append(dict(resource_id=rid, kind=kind, role=role, display_name=name, source_names=source_names,
                           turn_id=turn_id, image_id=image_id, group_id=group_id, bytes=path.stat().st_size,
                           status=status, sha256=digest, _path=path,
                           created_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                           preview_url=f'/api/sessions/{store.session_id}/resources/{rid}',
                           download_url=f'/api/sessions/{store.session_id}/resources/{rid}?download=true'))

    for image in images:
        add(image['image_path'], 'original', [image['source_name']], image_id=image['image_id'], digest=image['sha256'])
    turns_root = root / 'turns'
    if not turns_root.exists():
        return values
    for turn in sorted(turns_root.iterdir(), reverse=True):
        if not turn.is_dir():
            continue
        for manifest_path in turn.glob('artifacts/*/items/*/manifest.json'):
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                source = manifest['source']
                image_id = image_by_hash.get(source['sha256'])
                for entry in manifest['artifacts']:
                    if entry['role'] == 'original':
                        continue
                    add(manifest_path.parent / entry['path'], entry['role'], [source.get('source_name', '图片')],
                        turn.name, image_id, digest=entry['sha256'], status=manifest.get('status', 'running'))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        for manifest_path in turn.glob('groups/*/manifest.json'):
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                result = json.loads((manifest_path.parent / 'result.json').read_text(encoding='utf-8'))
                groups[result['group_id']] = {'turn_id': turn.name, 'image_ids': manifest['image_ids']}
                names = [item['source_name'] for item in result['per_image']]
                for entry in manifest['artifacts']:
                    add(manifest_path.parent / entry['path'], entry['role'], names, turn.name,
                        group_id=result['group_id'], digest=entry['sha256'], status=result['aggregate_status'])
            except (OSError, ValueError, KeyError, TypeError):
                continue
    for value in values:
        # 组筛选同时包括本轮分图成果与组内原图，不能只返回综合报告。
        value['group_ids'] = [gid for gid, group in groups.items() if value['group_id'] == gid or
            (value['image_id'] in group['image_ids'] and (value['turn_id'] is None or value['turn_id'] == group['turn_id']))]
    return values


def public_resource(value):
    return {key: item for key, item in value.items() if not key.startswith('_')}


def resolve_resource(store, resource_id):
    value = next((entry for entry in resource_index(store) if entry['resource_id'] == resource_id), None)
    if value is None:
        raise FileNotFoundError('资源不存在或未登记')
    if value['sha256'] and sha256(value['_path']) != value['sha256']:
        raise ValueError('资源内容与归档哈希不一致')
    return value
