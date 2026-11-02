"""只保存用户发送的附件；浏览本地图集不调用此服务。"""
import hashlib
import io
from pathlib import Path
import re
import uuid
from PIL import Image
from modules.agent.artifact_store import atomic_bytes


class UploadTooLarge(ValueError):
    """上传超过配置限制，HTTP适配层返回413。"""


async def receive_uploads(store, files, relative_paths, max_bytes, max_images):
    if not files or len(files) > max_images:
        raise ValueError('请选择不超过限制数量的图片')
    prepared = []
    for index, file in enumerate(files):
        source_name = str(file.filename or '').replace('\\', '/').split('/')[-1]
        if Path(source_name).suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            raise ValueError('只支持 JPG、PNG、BMP 航拍图片')
        chunks, size = [], 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise UploadTooLarge(f'{source_name} 超过单文件大小限制')
            chunks.append(chunk)
        data = b''.join(chunks)
        try:
            with Image.open(io.BytesIO(data)) as image:
                # RescueNet部分合法JPG带MPO元数据，Pillow将其识别为MPO，原分割入口可正常读取。
                if image.format not in {'JPEG', 'MPO', 'PNG', 'BMP'}:
                    raise ValueError('文件内容格式不受支持')
                image.verify()
        except Exception:
            raise ValueError(f'{source_name} 不是可读取的有效图片') from None
        digest = hashlib.sha256(data).hexdigest()
        if any(entry['sha256'] == digest for entry, _ in prepared):
            continue
        upload_id = 'upload_' + uuid.uuid4().hex
        safe_name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', source_name).strip(' .')
        safe_name = ('image' + Path(source_name).suffix) if not safe_name or len(safe_name) > 90 else safe_name
        path = store.session_dir / 'uploads' / upload_id / safe_name
        relative = relative_paths[index] if index < len(relative_paths) else source_name
        prepared.append((dict(upload_id=upload_id, path=str(path), source_name=source_name,
                              relative_path=str(relative), sha256=digest, bytes=size), data))
    # 整批验证成功才落盘，损坏的后续附件不会造成部分上传被当作完整提交。
    for entry, data in prepared:
        atomic_bytes(Path(entry['path']), data)
        store.save_upload(entry)
    return [{key: entry[key] for key in ['upload_id', 'source_name', 'relative_path', 'sha256', 'bytes']}
            for entry, _ in prepared]
