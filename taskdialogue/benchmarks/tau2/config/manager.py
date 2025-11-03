"""Configuration manager for Tau2-bench runs."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pathlib import Path

from taskdialogue.core.utils.logger import get_logger
from taskdialogue.benchmarks.tau2.config.defaults import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_ERRORS,
    DEFAULT_SEED,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_NUM_TRIALS,
)

logger = get_logger(__name__)


@dataclass
class RunConfig:
    """Configuration for a Tau2-bench run.
    
    This dataclass holds all parameters needed to run a Tau2 benchmark,
    extracted from the full TaskDialogue configuration.
    """
    # Domain settings
    domain: str = "airline"
    task_set_name: Optional[str] = None
    
    # Trial settings
    num_tasks: Optional[int] = None
    num_trials: int = DEFAULT_NUM_TRIALS
    
    # Limits
    max_steps: int = DEFAULT_MAX_STEPS
    max_errors: int = DEFAULT_MAX_ERRORS
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    
    # Reproducibility
    seed: int = DEFAULT_SEED
    
    # Tools
    tools_profile: str = "original"
    
    # Output
    save_to: Optional[Path] = None
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RunConfig":
        """Create RunConfig from TaskDialogue configuration dictionary.
        
        Args:
            config: Full TaskDialogue configuration with tau2 section
                    Expected format: {"tau2": {...}}
            
        Returns:
            RunConfig instance
        """
        # 获取嵌套字典中的值
        def get_nested(d: Dict[str, Any], keys: str, default: Any = None) -> Any:
            """从嵌套字典中获取值，支持点号分隔的键"""
            keys_list = keys.split('.')
            value = d
            for k in keys_list:
                if isinstance(value, dict):
                    value = value.get(k)
                    if value is None:
                        return default
                else:
                    return default
            return value
        
        # 从 config 中提取 tau2 部分
        tau2_config = config.get("tau2", {})
        
        return cls(
            domain=get_nested(tau2_config, "domain", "airline"),
            task_set_name=get_nested(tau2_config, "task_set"),
            num_tasks=get_nested(tau2_config, "trials.num_tasks"),
            num_trials=get_nested(tau2_config, "trials.num_trials", DEFAULT_NUM_TRIALS),
            max_steps=get_nested(tau2_config, "limits.max_steps", DEFAULT_MAX_STEPS),
            max_errors=get_nested(tau2_config, "limits.max_errors", DEFAULT_MAX_ERRORS),
            max_concurrency=get_nested(tau2_config, "max_concurrency", DEFAULT_MAX_CONCURRENCY),
            seed=get_nested(tau2_config, "seed", DEFAULT_SEED),
            tools_profile=get_nested(tau2_config, "tools.profile", "original"),
            save_to=get_nested(tau2_config, "save_to"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain": self.domain,
            "task_set_name": self.task_set_name,
            "num_tasks": self.num_tasks,
            "num_trials": self.num_trials,
            "max_steps": self.max_steps,
            "max_errors": self.max_errors,
            "max_concurrency": self.max_concurrency,
            "seed": self.seed,
            "tools_profile": self.tools_profile,
            "save_to": str(self.save_to) if self.save_to else None,
        }


class ConfigManager:
    """Manager for Tau2-bench configuration.
    
    Provides utilities for loading, validating, and merging configurations.
    """
    
    @staticmethod
    def merge_cli_args(config: Any, cli_args: Dict[str, Any]) -> Any:
        """Merge CLI arguments into configuration.
        
        Args:
            config: Base configuration (Config object or dict)
            cli_args: CLI arguments to override
            
        Returns:
            Merged configuration (same type as input)
        """
        # Config object doesn't need copying, it's mutable
        # Just modify in place and return
        merged = config
        
        # Map CLI args to config paths
        arg_mapping = {
            "domain": "tau2.domain",
            "num_tasks": "tau2.trials.num_tasks",
            "num_trials": "tau2.trials.num_trials",
            "max_steps": "tau2.limits.max_steps",
            "max_errors": "tau2.limits.max_errors",
            "seed": "tau2.seed",
            "tools_profile": "tau2.tools.profile",
            "provider": "model.agent.provider",
            "model": "model.agent.name",
            "temperature": "model.agent.temperature",
        }
        
        # Apply overrides
        for cli_key, config_path in arg_mapping.items():
            if cli_key in cli_args and cli_args[cli_key] is not None:
                merged.set(config_path, cli_args[cli_key])
        
        return merged
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default Tau2-bench configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "benchmark": {"type": "tau2"},
            "tau2": {
                "domain": "airline",
                "trials": {
                    "num_tasks": 5,
                    "num_trials": DEFAULT_NUM_TRIALS,
                },
                "limits": {
                    "max_steps": DEFAULT_MAX_STEPS,
                    "max_errors": DEFAULT_MAX_ERRORS,
                },
                "max_concurrency": DEFAULT_MAX_CONCURRENCY,
                "seed": DEFAULT_SEED,
                "tools": {
                    "profile": "original",
                },
            },
        }
    
    @staticmethod
    def print_config_summary(config: RunConfig) -> None:
        """Print a summary of the configuration.
        
        Args:
            config: RunConfig to summarize
        """
        logger.info("=" * 80)
        logger.info("Tau2-Bench Configuration")
        logger.info("=" * 80)
        logger.info(f"Domain:          {config.domain}")
        logger.info(f"Task Set:        {config.task_set_name or 'default'}")
        logger.info(f"Num Tasks:       {config.num_tasks or 'all'}")
        logger.info(f"Num Trials:      {config.num_trials}")
        logger.info(f"Max Steps:       {config.max_steps}")
        logger.info(f"Max Errors:      {config.max_errors}")
        logger.info(f"Concurrency:     {config.max_concurrency}")
        logger.info(f"Seed:            {config.seed}")
        logger.info(f"Tools Profile:   {config.tools_profile}")
        if config.save_to:
            logger.info(f"Save To:         {config.save_to}")
        logger.info("=" * 80)

