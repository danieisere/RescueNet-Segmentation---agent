"""
Configuration loader — reads settings from YAML.

Priority:
  1. Environment variable RERA_CONFIG (absolute path)
  2. ./config.yaml in project root
"""
import os
import yaml
from dotenv import load_dotenv


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=False)


def _find_config() -> str:
    """Locate the config YAML file."""
    env_path = os.environ.get('RERA_CONFIG')
    if env_path and os.path.isfile(env_path):
        return env_path

    # Default: config.yaml next to project root
    default = os.path.join(PROJECT_ROOT, 'config.yaml')
    if os.path.isfile(default):
        return default

    raise FileNotFoundError(
        "Cannot find config.yaml.\n"
        "  1. Copy config.yaml.example → config.yaml\n"
        "  2. Create .env from .env.example and fill in your LLM API key\n"
        "  Or set RERA_CONFIG env var to point to your config file."
    )


def load_config() -> dict:
    """Load and return the full configuration dictionary."""
    path = _find_config()
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# Load once at import time
_config = load_config()

# Export individual sections (backward-compatible with old agent_config.py)
LLM_CONFIG       = _config['llm']
REACT_CONFIG     = _config['react']
ENSEMBLE_CONFIG  = _config['ensemble']
PRIORITY_WEIGHTS = _config['priority_weights']
REPORT_CONFIG    = _config['report']
AGENT_CONFIG     = {'engine': 'langgraph', **_config.get('agent', {})}

# 旧的默认 reports 目录归一化；显式自定义目录继续作为产物根目录。
_legacy_output = REPORT_CONFIG.get('output_dir', './output/reports')
if os.path.normpath(_legacy_output) == os.path.normpath('./output/reports'):
    _legacy_output = './output/artifacts'
ARTIFACTS_CONFIG = {'output_dir': _legacy_output, **_config.get('artifacts', {})}
# 直接使用旧报告 API 时也遵循英文产物根目录，避免重建已迁移的 reports。
REPORT_CONFIG = {**REPORT_CONFIG, 'output_dir': ARTIFACTS_CONFIG['output_dir']}
GRAPH_CONFIG = {'max_steps': 50, 'intent_max_tokens': 1800, 'intent_max_attempts': 2,
                'intent_timeout': 90, **_config.get('graph', {})}

# 会话默认目录锚定项目根目录，切换终端工作目录也不会把记忆写到别处。
CONVERSATION_CONFIG = {'message_root': os.path.join(PROJECT_ROOT, 'message'),
                       'compression_threshold': 8000, 'recent_turns': 4,
                       'compression_target_tokens': 6000, 'compression_max_rounds': 4,
                       'summary_input_tokens': 12000, 'summary_max_batches': 32,
                       'model_context_window': 0,
                       'summary_max_tokens': 1800, 'intent_max_tokens': 2200,
                       'answer_max_tokens': 2400, **_config.get('conversation', {})}
if not os.path.isabs(CONVERSATION_CONFIG['message_root']):
    # 配置文件里的相对路径以项目为基准；CLI 的显式覆盖仍按用户当前目录解析。
    CONVERSATION_CONFIG['message_root'] = os.path.normpath(os.path.join(PROJECT_ROOT, CONVERSATION_CONFIG['message_root']))

# 综合分析使用独立输出预算；旧配置无需增加新节即可运行。
MULTIMODAL_CONFIG = {
    'max_tokens': 4096, 'temperature': 0.3, 'timeout': 180,
    'image_detail': 'auto', 'max_image_side': 1536, 'max_attempts': 2,
    **_config.get('multimodal', {}),
}


def get_llm_api_key() -> str:
    """Resolve the LLM API key from config or a configured environment variable."""
    configured_key = LLM_CONFIG.get('api_key')
    if configured_key and configured_key != 'YOUR_DEEPSEEK_API_KEY':
        return configured_key

    api_key_env = LLM_CONFIG.get('api_key_env')
    env_names = [api_key_env, 'DEEPSEEK_API_KEY', 'OPENAI_API_KEY']
    for env_name in env_names:
        if env_name and os.environ.get(env_name):
            return os.environ[env_name]

    return ''
