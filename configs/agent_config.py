"""
Agent configuration — re-exports from config.yaml for backward compatibility.

All existing imports keep working:
    from configs.agent_config import LLM_CONFIG, REACT_CONFIG, ...

To change settings, edit the project-root config.yaml file.
"""
from configs.config_loader import (
    LLM_CONFIG,
    REACT_CONFIG,
    ENSEMBLE_CONFIG,
    PRIORITY_WEIGHTS,
    REPORT_CONFIG,
    AGENT_CONFIG,
)
