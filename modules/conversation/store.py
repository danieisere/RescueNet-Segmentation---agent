"""SQLite 会话记忆及可重建审计档案，业务数据以数据库为准。"""

from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterator
import uuid


_TZ = timezone(timedelta(hours=8))
_SESSION_ID = re.compile(r"\d{8}_\d{6}\Z")
_TURN_ID = re.compile(r"\d{6,}\Z")


def _now() -> str:
    return datetime.now(_TZ).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    # 不使用 default=str，防止数组、模型等运行时对象被悄悄持久化。
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _atomic_write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content.encode("utf-8") if isinstance(content, str) else content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class SessionStore:
    """同一会话由控制器持有 lock；每个方法使用短事务和独立连接。"""

    def __init__(self, root: str | Path, session_id: str):
        if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
            raise ValueError("会话 ID 必须采用 YYYYMMDD_HHMMSS 格式")
        try:
            datetime.strptime(session_id, "%Y%m%d_%H%M%S")
        except ValueError as exc:
            raise ValueError("会话 ID 包含无效日期") from exc
        self.root = Path(root).expanduser().resolve()
        self.session_id = session_id
        self.session_dir = (self.root / session_id).resolve()
        if self.session_dir.parent != self.root:
            raise ValueError("会话目录不能超出根目录")
        self.db_path = self.session_dir / "memory.sqlite"
        if not self.session_dir.is_dir() or not self.db_path.is_file():
            raise FileNotFoundError(f"会话不存在或缺少数据库：{session_id}")

    @classmethod
    def create(cls, root: str | Path) -> "SessionStore":
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        while True:
            session_id = datetime.now(_TZ).strftime("%Y%m%d_%H%M%S")
            directory = root / session_id
            try:
                # mkdir 是跨进程原子占位；碰撞后等待真实时间变化，不能伪造未来秒数。
                directory.mkdir()
                break
            except FileExistsError:
                time.sleep(0.05)
        database = directory / "memory.sqlite"
        # SQLite 的连接上下文仅提交事务，不关闭句柄，因此显式 closing。
        with closing(sqlite3.connect(database)) as connection, connection:
            connection.executescript("""
                CREATE TABLE meta (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL);
                CREATE TABLE turns (turn_id TEXT PRIMARY KEY, request_id TEXT UNIQUE, payload TEXT NOT NULL);
                CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, turn_id TEXT NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL, timestamp TEXT NOT NULL,
                    status TEXT NOT NULL, UNIQUE(turn_id, role));
                CREATE TABLE images (image_id TEXT PRIMARY KEY, sha256 TEXT UNIQUE NOT NULL,
                    payload TEXT NOT NULL);
                CREATE TABLE facts (fact_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE operations (turn_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE summaries (through_message_id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE analyses (id INTEGER PRIMARY KEY AUTOINCREMENT, turn_id TEXT NOT NULL,
                    image_id TEXT, payload TEXT NOT NULL, UNIQUE(turn_id, image_id));
            """)
            connection.execute("INSERT INTO meta VALUES (1, ?)", (_json({
                "session_id": session_id, "created_at": _now(), "version": 0,
                "current_image_id": None, "current_image_path": None, "current_focus": [],
                "pending_clarification": None, "last_turn": 0,
            }),))
        store = cls(root, session_id)
        store.export()
        return store

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            # 多表写入以同一短事务提交；异常时事实更正和版本号全部回滚。
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _meta(connection: sqlite3.Connection) -> dict:
        return json.loads(connection.execute("SELECT payload FROM meta WHERE id=1").fetchone()[0])

    @staticmethod
    def _put_meta(connection: sqlite3.Connection, meta: dict) -> None:
        connection.execute("UPDATE meta SET payload=? WHERE id=1", (_json(meta),))

    def meta(self) -> dict:
        with self._db() as connection:
            return self._meta(connection)

    def update_meta(self, **fields: Any) -> dict:
        # 图片必须经 attach_image 验证入库，编号和路径不能由外部随意覆盖。
        if set(fields) - {"current_focus", "pending_clarification"}:
            raise ValueError("仅允许更新 current_focus / pending_clarification")
        if "current_focus" in fields:
            topics = {"all", "buildings", "roads", "water", "vehicles", "damage", "rescue",
                      "vegetation", "pool", "background"}
            focus = fields["current_focus"]
            # 与 TaskSpec 保持相同类别协议；bool 属于 int 子类，必须单独排除。
            if not isinstance(focus, list) or any(
                not ((type(item) is int and 0 <= item <= 10)
                     or (isinstance(item, str) and item in topics)) for item in focus
            ):
                raise ValueError("current_focus 必须是已知关注主题或 0 至 10 类别编号的列表")
        _json(fields)
        with self._db() as connection:
            meta = self._meta(connection)
            if any(meta[key] != value for key, value in fields.items()):
                meta.update(fields)
                meta["version"] += 1
                self._put_meta(connection, meta)
        self.export()
        return meta

    @contextmanager
    def lock(self) -> Iterator["SessionStore"]:
        lock_path = self.session_dir / ".session.lock"
        with lock_path.open("a+b") as stream:
            if lock_path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("当前会话正在由其他请求使用") from exc
            try:
                yield self
            finally:
                # OS 锁随进程退出释放；文件存在本身不代表会话忙碌。
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _turn(self, connection: sqlite3.Connection, turn_id: str) -> dict:
        if not isinstance(turn_id, str) or not _TURN_ID.fullmatch(turn_id):
            raise ValueError("无效轮次编号")
        row = connection.execute("SELECT payload FROM turns WHERE turn_id=?", (turn_id,)).fetchone()
        if row is None:
            raise ValueError("轮次不存在")
        return json.loads(row[0])

    def get_turn(self, turn_id: str) -> dict:
        """直接读取权威终态，供文件导出失败后的控制器核对数据库。"""
        with self._db() as connection:
            return self._turn(connection, turn_id)

    def begin_turn(self, question: str, request_id: str | None = None,
                   input_metadata: dict | None = None) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("问题不能为空")
        if request_id is not None and (not isinstance(request_id, str) or not request_id.strip()):
            raise ValueError("request_id 必须是非空字符串")
        if input_metadata is None:
            input_metadata = {}
        if not isinstance(input_metadata, dict):
            raise ValueError("输入附件元数据必须是 JSON 对象")
        input_metadata = json.loads(_json(input_metadata))
        with self._db() as connection:
            row = connection.execute("SELECT payload FROM turns WHERE request_id=?", (request_id,)).fetchone()
            if row is not None:
                turn = json.loads(row[0])
                # 未完成轮次必须继续同一份输入，否则新事实会错误引用旧用户原文。
                # 完成轮次直接重放已有结果，保持 request_id 原有幂等语义。
                if "answer" not in turn and (
                    question != turn["question"] or input_metadata != turn.get("input_metadata", {})
                ):
                    raise ValueError("未完成请求的原文或附件已改变，请使用新的 request_id")
            else:
                meta = self._meta(connection)
                meta["last_turn"] += 1
                turn_id = f"{meta['last_turn']:06d}"
                timestamp = _now()
                cursor = connection.execute(
                    "INSERT INTO messages(turn_id,role,content,timestamp,status) VALUES (?, 'user', ?, ?, 'running')",
                    (turn_id, question, timestamp),
                )
                turn = {"turn_id": turn_id, "turn_dir": str(self.session_dir / "turns" / turn_id),
                        "question": question, "status": "running", "user_message_id": cursor.lastrowid,
                        "request_id": request_id, "created_at": timestamp,
                        "input_metadata": input_metadata}
                connection.execute("INSERT INTO turns VALUES (?, ?, ?)", (turn_id, request_id, _json(turn)))
                self._put_meta(connection, meta)
        self.export()
        return turn

    def mark_interrupted(self) -> None:
        with self._db() as connection:
            for row in connection.execute("SELECT turn_id,payload FROM turns").fetchall():
                turn = json.loads(row["payload"])
                if turn["status"] == "running":
                    turn.update(status="interrupted", finished_at=_now())
                    connection.execute("UPDATE turns SET payload=? WHERE turn_id=?", (_json(turn), row["turn_id"]))
                    connection.execute("UPDATE messages SET status='interrupted' WHERE turn_id=?", (row["turn_id"],))
        self.export()

    def finish_turn(self, turn_id: str, answer: str, status: str = "completed", result: dict | None = None) -> dict:
        if status not in {"completed", "failed", "interrupted", "needs_clarification", "unsupported"}:
            raise ValueError("不支持的轮次终态")
        if not isinstance(answer, str):
            raise ValueError("回答必须是文本")
        _json(result)
        with self._db() as connection:
            turn = self._turn(connection, turn_id)
            # 重试不能改写已归档的历史回答，也不能重复插入 assistant 消息。
            if "answer" not in turn:
                timestamp = _now()
                turn.update(answer=answer, status=status, result=result, finished_at=timestamp)
                connection.execute("UPDATE turns SET payload=? WHERE turn_id=?", (_json(turn), turn_id))
                connection.execute("UPDATE messages SET status=? WHERE turn_id=?", (status, turn_id))
                connection.execute(
                    "INSERT INTO messages(turn_id,role,content,timestamp,status) VALUES (?, 'assistant', ?, ?, ?)",
                    (turn_id, answer, timestamp, status),
                )
        self.export()
        return turn

    def record_archive_failure(self, turn_id: str, error: str) -> dict:
        """只修正归档终态，保留已经产生的回答和部分成果引用。"""
        if not isinstance(error, str) or not error.strip():
            raise ValueError("归档失败必须提供非空错误说明")
        with self._db() as connection:
            turn = self._turn(connection, turn_id)
            if "answer" not in turn:
                raise ValueError("尚未提交回答的轮次不能使用归档失败修正")
            # 数据库先于文件提交，永久写盘故障时也不能留下 completed 的假终态。
            # 这是终态提交异常的专用入口，普通 finish_turn 仍保留原有幂等语义。
            result = dict(turn.get("result") or {})
            result.update(status="failed", error={"stage": "archive", "message": error})
            turn.update(status="failed", result=result, archive_error=error)
            connection.execute("UPDATE turns SET payload=? WHERE turn_id=?", (_json(turn), turn_id))
            connection.execute("UPDATE messages SET status='failed' WHERE turn_id=?", (turn_id,))
        # 即使这里仍失败，以上 failed 状态已经提交；控制器通过 get_turn 读取。
        self.export()
        return turn

    def messages(self) -> list[dict]:
        with self._db() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM messages ORDER BY id")]

    def message_image_ids(self) -> dict[int, str | None]:
        """从附件哈希与真实分析记录恢复消息范围，兼容未存 image_id 的旧会话。"""
        with self._db() as connection:
            images = {row['sha256']: row['image_id'] for row in connection.execute('SELECT image_id,sha256 FROM images')}
            analyses = {row['turn_id']: row['image_id'] for row in connection.execute('SELECT turn_id,image_id FROM analyses')}
            scopes, current = {}, None
            for row in connection.execute('SELECT turn_id,payload FROM turns ORDER BY turn_id'):
                turn = json.loads(row['payload'])
                digest = turn.get('input_metadata', {}).get('image_sha256')
                current = images.get(digest, analyses.get(row['turn_id'], current))
                scopes[row['turn_id']] = current
            return {row['id']: scopes.get(row['turn_id']) for row in connection.execute('SELECT id,turn_id FROM messages')}

    def attach_image(self, path: str | Path) -> dict:
        from PIL import Image
        import io
        path = Path(path).expanduser().resolve()
        data = path.read_bytes()
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
        except Exception as exc:
            raise ValueError("输入不是有效图像") from exc
        digest = hashlib.sha256(data).hexdigest()
        with self._db() as connection:
            row = connection.execute("SELECT payload FROM images WHERE sha256=?", (digest,)).fetchone()
            if row is not None:
                saved = json.loads(row[0])
                if not saved.get("source_name"):
                    # 旧会话可能只有 source_path；重新绑定同一图片时补齐显示名，内部路径不变。
                    legacy_path = saved.get("source_path") or saved.get("image_path") or ""
                    saved["source_name"] = Path(legacy_path).name or path.name
                    connection.execute("UPDATE images SET payload=? WHERE sha256=?", (_json(saved), digest))
            else:
                number = connection.execute("SELECT COUNT(*) FROM images").fetchone()[0] + 1
                image_id = f"image_{number:04d}"
                extension = path.suffix.lower()
                if not re.fullmatch(r"\.[a-z0-9]{1,10}", extension):
                    extension = ".img"
                destination = self.session_dir / "inputs" / image_id / f"original{extension}"
                # 保存原始字节，避免 RGB 转换或 JPEG 重编码改变输入证据。
                _atomic_write(destination, data)
                saved = {"image_id": image_id, "image_path": str(destination), "sha256": digest,
                         # source_name 只用于用户可见显示；内部仍使用 image_id 和 image_path。
                         "source_path": str(path), "source_name": path.name, "created_at": _now()}
                connection.execute("INSERT INTO images VALUES (?, ?, ?)", (image_id, digest, _json(saved)))
            meta = self._meta(connection)
            if meta["current_image_id"] != saved["image_id"]:
                meta.update(current_image_id=saved["image_id"], current_image_path=saved["image_path"])
                meta["version"] += 1
                self._put_meta(connection, meta)
        self.export()
        return saved

    def image_metadata(self, image_id: str) -> dict | None:
        """读取图片元数据，供控制器向分析图传递稳定的源文件显示名。"""
        with self._db() as connection:
            row = connection.execute("SELECT payload FROM images WHERE image_id=?", (image_id,)).fetchone()
        return json.loads(row[0]) if row is not None else None

    def facts(self, active_only: bool = True, image_id: str | None = None) -> list[dict]:
        with self._db() as connection:
            facts = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM facts ORDER BY fact_id")]
        return [fact for fact in facts if (not active_only or fact["status"] == "active")
                and (image_id is None or fact["scope"] == "session" or fact["image_id"] == image_id)]

    def apply_changes(self, turn_id: str, changes: list[dict], image_id: str | None = None) -> list[dict]:
        if not isinstance(changes, list):
            raise ValueError("事实变更必须是列表")
        with self._db() as connection:
            turn = self._turn(connection, turn_id)
            operation = connection.execute("SELECT payload FROM operations WHERE turn_id=?", (turn_id,)).fetchone()
            if operation is not None:
                return json.loads(operation[0])
            if image_id is not None and connection.execute(
                "SELECT 1 FROM images WHERE image_id=?", (image_id,)
            ).fetchone() is None:
                raise ValueError("图片尚未登记")
            output, changed = [], False
            for index, raw in enumerate(changes, 1):
                allowed = {"operation", "key", "value", "unit", "location", "scope", "supersedes_fact_id", "verification", "observed_at"}
                if not isinstance(raw, dict) or set(raw) - allowed:
                    raise ValueError("事实变更字段非法")
                if raw.get("operation") not in {"add", "correct"}:
                    raise ValueError("事实操作必须为 add 或 correct")
                if any(not isinstance(raw.get(key), str) or not raw[key].strip() for key in ("key", "value")):
                    raise ValueError("事实必须包含非空 key/value")
                scope = raw.get("scope", "image")
                if scope not in {"image", "session"} or (scope == "image" and not image_id):
                    raise ValueError("图片事实缺少图片范围，需要先澄清")
                verification = raw.get("verification", "user_reported")
                if verification not in {"user_reported", "confirmed", "unverified"}:
                    raise ValueError("无效核实状态")
                for key in ("unit", "location", "observed_at"):
                    if not isinstance(raw.get(key, ""), str):
                        raise ValueError(f"{key} 必须是文本")
                target_image = image_id if scope == "image" else None
                location = raw.get("location", "")
                active = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM facts")]
                candidates = [fact for fact in active if fact["status"] == "active"
                              and (fact["key"], fact["location"], fact["scope"], fact["image_id"])
                              == (raw["key"], location, scope, target_image)]
                old = None
                if raw["operation"] == "add":
                    if raw.get("supersedes_fact_id") is not None:
                        raise ValueError("新增事实不能指定替代目标")
                    if candidates:
                        if len(candidates) == 1 and candidates[0]["value"] == raw["value"] and candidates[0]["unit"] == raw.get("unit", ""):
                            output.append(candidates[0])
                            continue
                        raise ValueError("同范围已有冲突事实，请明确更正")
                else:
                    target = raw.get("supersedes_fact_id")
                    matches = [fact for fact in candidates if target is None or fact["fact_id"] == target]
                    if len(matches) != 1:
                        raise ValueError("更正目标缺失或不唯一，需要澄清")
                    old = matches[0]
                fact = {"fact_id": f"{turn_id}_{index:03d}", "key": raw["key"], "value": raw["value"],
                        "unit": raw.get("unit", ""), "location": location, "scope": scope,
                        "image_id": target_image, "status": "active", "source": "user_report",
                        "source_message_id": turn["user_message_id"], "turn_id": turn_id,
                        "created_at": _now(), "observed_at": raw.get("observed_at", ""),
                        "verification": verification, "supersedes_fact_id": old["fact_id"] if old else None}
                if old:
                    old.update(status="superseded", superseded_by_fact_id=fact["fact_id"])
                    connection.execute("UPDATE facts SET payload=? WHERE fact_id=?", (_json(old), old["fact_id"]))
                connection.execute("INSERT INTO facts VALUES (?, ?)", (fact["fact_id"], _json(fact)))
                output.append(fact)
                changed = True
            connection.execute("INSERT INTO operations VALUES (?, ?)", (turn_id, _json(output)))
            if changed:
                meta = self._meta(connection)
                meta["version"] += 1
                self._put_meta(connection, meta)
        self.export()
        return output

    def summaries(self) -> list[dict]:
        with self._db() as connection:
            return [json.loads(row[0]) for row in connection.execute("SELECT payload FROM summaries ORDER BY through_message_id")]

    def save_summary(self, content: dict, through_message_id: int, refine: bool = False) -> dict:
        if not isinstance(content, dict) or not content:
            raise ValueError("摘要必须是非空对象")
        _json(content)
        with self._db() as connection:
            row = connection.execute("SELECT payload FROM summaries WHERE through_message_id=?", (through_message_id,)).fetchone()
            if row is not None:
                summary = json.loads(row[0])
                if refine and summary['content'] != content:
                    # 同一覆盖边界可进一步压缩；旧有效版本仍作为审计记录保留，失败压缩不调用本方法。
                    revisions = list(summary.get('revisions', []))
                    revisions.append({'content': summary['content'], 'created_at': summary['created_at']})
                    summary.update(content=content, created_at=_now(), revisions=revisions)
                    connection.execute('UPDATE summaries SET payload=? WHERE through_message_id=?',
                                       (_json(summary), through_message_id))
            else:
                if type(through_message_id) is not int or connection.execute(
                    "SELECT 1 FROM messages WHERE id=?", (through_message_id,)
                ).fetchone() is None:
                    raise ValueError("摘要边界消息不存在")
                summary = {"content": content, "through_message_id": through_message_id, "created_at": _now()}
                connection.execute("INSERT INTO summaries VALUES (?, ?)", (through_message_id, _json(summary)))
        self.export()
        return summary

    def register_analysis(self, turn_id: str, image_id: str | None, result: dict) -> None:
        if not isinstance(result, dict):
            raise ValueError("分析索引必须是结构化结果")
        _json(result)
        with self._db() as connection:
            self._turn(connection, turn_id)
            record = {"turn_id": turn_id, "image_id": image_id, "item_dir": result.get("item_dir"),
                      "result": result, "created_at": _now()}
            # NULL 不受 SQLite UNIQUE 约束，因此显式查询确保无图片结果也幂等。
            existing = connection.execute("SELECT id FROM analyses WHERE turn_id=? AND image_id IS ?", (turn_id, image_id)).fetchone()
            if existing is None:
                connection.execute("INSERT INTO analyses(turn_id,image_id,payload) VALUES (?, ?, ?)", (turn_id, image_id, _json(record)))

    def analyses(self, image_id: str | None) -> list[dict]:
        with self._db() as connection:
            return [json.loads(row[0]) for row in connection.execute(
                "SELECT payload FROM analyses WHERE image_id IS ? ORDER BY id DESC", (image_id,)
            )]

    def export(self) -> None:
        # 单次读快照保证各导出来自同一数据库版本；失败后仍可再次重建。
        with self._db() as connection:
            meta = self._meta(connection)
            messages = [dict(row) for row in connection.execute("SELECT * FROM messages ORDER BY id")]
            facts = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM facts ORDER BY fact_id")]
            summaries = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM summaries ORDER BY through_message_id")]
            turns = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM turns ORDER BY turn_id")]
            images = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM images ORDER BY image_id")]
        _atomic_write(self.session_dir / "session.json", _json(meta))
        _atomic_write(self.session_dir / "messages.jsonl", "".join(_json(message) + "\n" for message in messages))
        _atomic_write(self.session_dir / "facts.json", _json(facts))
        _atomic_write(self.session_dir / "summaries.json", _json(summaries))
        for turn in turns:
            directory = self.session_dir / "turns" / turn["turn_id"]
            request = {key: value for key, value in turn.items() if key not in {"answer", "result", "finished_at"}}
            _atomic_write(directory / "request.json", _json(request))
            if turn["status"] != "running":
                _atomic_write(directory / "result.json", _json(turn))
        for image in images:
            _atomic_write(self.session_dir / "inputs" / image["image_id"] / "metadata.json", _json(image))

    @classmethod
    def list_sessions(cls, root: str | Path) -> list[dict]:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            return []
        sessions = []
        for directory in sorted(root.iterdir(), reverse=True):
            if directory.is_dir() and _SESSION_ID.fullmatch(directory.name):
                try:
                    sessions.append(cls(root, directory.name).meta())
                except (ValueError, FileNotFoundError, sqlite3.DatabaseError):
                    # 原子占位后尚未建库或损坏的目录不能伪装成有效会话。
                    continue
        return sessions
