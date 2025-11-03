"""
Configuration management for TaskDialogue.

Provides a clean interface for loading and accessing configuration from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Configuration container with nested key access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot-separated keys.
        
        Args:
            key: Configuration key (e.g., 'model.temperature')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            >>> config.get('model.temperature', 0.7)
            0.1
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        # Handle environment variable substitution
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.getenv(env_var, default)
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot-separated keys.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting."""
        self.set(key, value)
    
    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default.
        
    Returns:
        Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        
    Example:
        >>> config = load_config('configs/my_config.yaml')
        >>> model_name = config.get('model.name')
    """
    if config_path is None:
        # Use default configuration
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "default.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f) or {}
    
    return Config(config_dict)


def get_api_key(provider: str, config: Optional[Config] = None) -> Optional[str]:
    """
    Get API key for a provider.
    
    Priority:
    1. Config file (api.{provider}.api_key)
    2. Environment variable ({PROVIDER}_API_KEY)
    
    Args:
        provider: Provider name ('openai', 'deepseek', 'zhipuai', etc.)
        config: Config object (optional)
        
    Returns:
        API key or None
        
    Example:
        >>> api_key = get_api_key('openai')
        >>> print(api_key)
        'sk-...'
    """
    # Try config file first
    if config is not None:
        key = config.get(f'api.{provider}.api_key')
        if key:
            return key
    
    # Try environment variable
    env_var = f'{provider.upper()}_API_KEY'
    return os.getenv(env_var)


def save_config(config: Config, output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            config.to_dict(),
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False
        )

