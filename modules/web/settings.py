"""YAML白名单配置编辑，保留注释与未知键；运行时快照只在重启后变化。"""
from copy import deepcopy
import hashlib
import io
import math
import os
from pathlib import Path
import threading
from ruamel.yaml import YAML

from configs.config_loader import PROJECT_ROOT, CONVERSATION_CONFIG
from modules.agent.artifact_store import atomic_bytes
from .jobs import JobConflict

EDITABLE = {
    'llm': {'provider', 'model', 'base_url', 'wire_api', 'reasoning_effort', 'verbosity'},
    'ensemble': {'models', 'confidence_lambda'},
    'priority_weights': {'destroyed', 'major', 'minor', 'road_blocked'},
    # 旧top_k_priority_zones未被现有报告路径消费，因此不伪装成有效开关。
    'report': {'format', 'include_visualization'},
    'agent': {'device'}, 'vision': {'max_input_images'}, 'web': {'max_upload_bytes'},
}
DEFAULTS = {'vision': {'max_input_images': 5}, 'web': {'max_upload_bytes': 20 * 1024 * 1024},
            'conversation': {key: CONVERSATION_CONFIG[key] for key in
                             ['compression_threshold', 'compression_target_tokens', 'recent_turns']}}


class SettingsService:
    def __init__(self, path, runtime_config=None):
        self.path = Path(path).resolve()
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.lock = threading.Lock()
        self.started_hash = self._version()
        self.runtime = deepcopy(runtime_config if runtime_config is not None else self._load())

    def _version(self):
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def _load(self):
        return self.yaml.load(self.path.read_text(encoding='utf-8')) or {}

    @property
    def restart_required(self):
        return self._version() != self.started_hash

    @staticmethod
    def public(config):
        # 显式选择字段，不能靠字符串替换遮掩凭据或把未知键传到浏览器。
        values = {section: {key: deepcopy(value) for key, value in config.get(section, {}).items() if key in fields}
                  for section, fields in EDITABLE.items()}
        for section, fields in DEFAULTS.items():
            if section != 'conversation':
                values[section] = {**fields, **values.get(section, {})}
        return values

    def read(self):
        with self.lock:
            data = self._load()
            env_name = data.get('llm', {}).get('api_key_env', 'OPENAI_API_KEY')
            configured = bool(data.get('llm', {}).get('api_key')) or any(
                bool(os.environ.get(name)) for name in [env_name, 'DEEPSEEK_API_KEY', 'OPENAI_API_KEY'] if isinstance(name, str))
            return dict(version=self._version(), values=self.public(data), runtime=self.public(self.runtime),
                        defaults=DEFAULTS, restart_required=self.restart_required, editable={k: sorted(v) for k,v in EDITABLE.items()},
                        source=str(self.path), credentials={'configured': configured, 'api_key_env': env_name},
                        readonly={'compression_threshold': CONVERSATION_CONFIG['compression_threshold'],
                                  'compression_target_tokens': CONVERSATION_CONFIG['compression_target_tokens'],
                                  'recent_turns': CONVERSATION_CONFIG['recent_turns'],
                                  'weight_dir': data.get('ensemble', {}).get('weight_dir', './runs'),
                                  'message_root': CONVERSATION_CONFIG['message_root'], 'engine': 'langgraph',
                                  'report_directory': 'message/<session_id>/turns/<turn_id>/'})

    def _validate(self, section, key, value, data):
        if key in {'confidence_lambda', 'destroyed', 'major', 'minor', 'road_blocked'}:
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise ValueError('权重必须为非负有限数值')
        elif key in {'max_input_images', 'max_upload_bytes', 'top_k_priority_zones'}:
            if type(value) is not int or value < 1:
                raise ValueError('数量和大小限制必须为正整数')
            if key == 'max_input_images' and value > 100:
                raise ValueError('单轮图片上限不能超过100')
        elif key == 'include_visualization':
            if type(value) is not bool:
                raise ValueError('可视化开关必须是布尔值')
        elif key == 'models':
            if not isinstance(value, list) or not value or any(not isinstance(name, str) for name in value) or len(set(value)) != len(value) or any(name not in {'unet','pspnet','deeplabv3'} for name in value):
                raise ValueError('请选择有效且不重复的分割模型')
            weight_dir = Path(data.get('ensemble', {}).get('weight_dir', './runs'))
            weight_dir = weight_dir if weight_dir.is_absolute() else Path(PROJECT_ROOT)/weight_dir
            if any(not (weight_dir / (name + '_final.pth')).is_file() for name in value):
                raise ValueError('所选分割模型缺少权重文件')
        else:
            if not isinstance(value, str) or not value.strip():
                raise ValueError('设置内容不能为空')
            options = {'format': {'html','markdown'}, 'wire_api': {'responses','chat_completions'},
                       'verbosity': {'low','medium','high'}, 'reasoning_effort': {'none','minimal','low','medium','high','xhigh'},
                       'device': {'auto','cpu','cuda'}}
            if key in options and value not in options[key]:
                raise ValueError('不支持的设置值：' + key)
            if key == 'base_url' and not value.startswith(('http://','https://')):
                raise ValueError('接口地址必须以http://或https://开头')
            if key == 'device' and value == 'cuda':
                import torch
                if not torch.cuda.is_available():
                    raise ValueError('当前环境未检测到可用CUDA设备')

    def save(self, version, changes):
        with self.lock:
            if self._version() != version:
                raise JobConflict('配置已由其他页面或程序修改，请重新打开设置后保存')
            data = self._load()
            for section, fields in changes.items():
                if section not in EDITABLE or not isinstance(fields, dict) or set(fields) - EDITABLE[section]:
                    raise ValueError('包含不可编辑的设置字段')
                for key, value in fields.items():
                    self._validate(section,key,value,data)
            for section, fields in changes.items():
                if fields:
                    data.setdefault(section, {}).update(fields)
            output = io.StringIO()
            self.yaml.dump(data,output)
            # 原子替换前再次核对，避免把外部编辑覆盖为旧快照。
            if self._version() != version:
                raise JobConflict('保存期间配置文件发生变化，请重新加载')
            atomic_bytes(self.path,output.getvalue().encode('utf-8'))
        return self.read()
