"""多轮会话接口；保持轻量，避免存储访问触发模型加载。"""

from .store import SessionStore

__all__ = ["SessionStore"]
