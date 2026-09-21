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
AGENT_CONFIG     = _config['agent']


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
