"""历史图片缺少网页显示字段时，读取、资源访问和附件复用仍应正常。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from modules.conversation.store import SessionStore
from modules.web.resources import resource_index


HAS_WEB_DEPS = all(importlib.util.find_spec(name) for name in ('fastapi', 'httpx'))


class LegacyImageMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = SessionStore.create(self.root)
        self.image_path = self.store.session_dir / 'inputs' / 'image_0001' / 'original.jpg'
        self.image_path.parent.mkdir(parents=True)
        # 仅测试资源归档和哈希校验，不触发模型推理或图片重新编码。
        self.image_path.write_bytes(b'archived-image-evidence')
        self.legacy = dict(image_id='image_0001', image_path=str(self.image_path),
                           sha256=hashlib.sha256(self.image_path.read_bytes()).hexdigest(),
                           source_path=r'D:\legacy\aerial\10794.jpg')
        self.save_record(self.legacy)

    def save_record(self, record):
        with sqlite3.connect(self.store.db_path) as connection:
            connection.execute('INSERT OR REPLACE INTO images VALUES (?, ?, ?)',
                               (record['image_id'], record['sha256'], json.dumps(record)))

    def snapshot(self):
        # 比较全部会话文件，保证显示字段兼容不通过迁移回写历史数据。
        return {path.relative_to(self.store.session_dir).as_posix():
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.store.session_dir.rglob('*') if path.is_file()}

    def test_legacy_names_are_available_without_rewriting_history(self):
        before = self.snapshot()
        listed = self.store.images()[0]
        selected = self.store.image_metadata('image_0001')
        self.assertEqual(listed.get('source_name'), '10794.jpg')
        self.assertEqual(listed.get('relative_path'), '10794.jpg')
        self.assertEqual(selected, listed)
        self.assertEqual(listed['image_path'], self.legacy['image_path'])
        self.assertEqual(listed['sha256'], self.legacy['sha256'])
        self.assertEqual(self.snapshot(), before)
        self.assertIsNone(self.store.image_metadata('missing_image'))

    def test_missing_source_path_falls_back_to_archived_filename(self):
        record = {key: value for key, value in self.legacy.items() if key != 'source_path'}
        self.save_record(record)
        self.assertEqual(self.store.images()[0].get('source_name'), 'original.jpg')

    def test_current_names_and_relative_paths_are_preserved(self):
        record = dict(self.legacy, source_name='航拍现场.jpg', relative_path='灾区甲/航拍现场.jpg')
        self.save_record(record)
        self.assertEqual(self.store.images()[0], record)
        self.assertEqual(self.store.image_metadata('image_0001'), record)

    def test_legacy_original_is_present_in_resource_index(self):
        before = self.snapshot()
        resources = resource_index(self.store)
        self.assertEqual(len(resources), 1)
        self.assertEqual(resources[0]['source_names'], ['10794.jpg'])
        self.assertEqual(resources[0]['image_id'], 'image_0001')
        self.assertEqual(self.snapshot(), before)

    @unittest.skipUnless(HAS_WEB_DEPS, '接口验证需要可选网页依赖')
    def test_legacy_session_api_download_and_archived_attachment(self):
        from fastapi.testclient import TestClient
        from modules.web.api import create_app
        from modules.web.schemas import ImageRef

        before = self.snapshot()
        app = create_app(message_root=self.root)
        with TestClient(app) as client:
            prefix = f'/api/sessions/{self.store.session_id}'
            session = client.get(prefix)
            self.assertEqual(session.status_code, 200)
            self.assertEqual(session.json()['images'][0]['source_name'], '10794.jpg')
            response = client.get(prefix + '/resources')
            self.assertEqual(response.status_code, 200)
            resource = response.json()['resources'][0]
            for url in [resource['preview_url'], resource['download_url']]:
                downloaded = client.get(url)
                self.assertEqual(downloaded.status_code, 200)
                self.assertEqual(downloaded.content, self.image_path.read_bytes())
            self.assertEqual(client.get(prefix + '/messages').status_code, 200)
            # 只解析归档附件，验证继续分析的入口，不提交任务或调用大模型。
            sources = app.state.manager._sources(self.store, [ImageRef(kind='archived', id='image_0001')])
            self.assertEqual(sources[0]['source_name'], '10794.jpg')
            self.assertEqual(sources[0]['relative_path'], '10794.jpg')
        self.assertEqual(self.snapshot(), before)


if __name__ == '__main__':
    unittest.main()
