"""
Configuration loader — reads settings from YAML.

Priority:
  1. Environment variable RERA_CONFIG (absolute path)
  2. ./config.yaml in project root
"""
import os
import yaml


def _find_config() -> str:
    """Locate the config YAML file."""
    env_path = os.environ.get('RERA_CONFIG')
    if env_path and os.path.isfile(env_path):
        return env_path

    # Default: config.yaml next to project root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default = os.path.join(root, 'config.yaml')
    if os.path.isfile(default):
        return default

    raise FileNotFoundError(
        "Cannot find config.yaml.\n"
        "  1. Copy config.yaml.example → config.yaml\n"
        "  2. Fill in your DeepSeek API key\n"
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
