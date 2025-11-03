"""Unified logging system for TaskDialogue.

This module provides a centralized logging configuration that:
- Initializes from YAML configuration
- Supports multiple log levels and formats
- Works across all modules (taskdialogue, tau2_bench, etc.)
- No dependency on .env files

Usage:
    # In your module
    from taskdialogue.utils.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Starting process...")
    
    # Configure from config
    from taskdialogue.utils.config import load_config
    from taskdialogue.utils.logger import setup_logging
    
    config = load_config("configs/default.yaml")
    setup_logging(config)
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


# Default configuration
DEFAULT_LOG_CONFIG = {
    "level": "INFO",
    "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    "colorize": True,
    "backtrace": True,
    "diagnose": True,
}

# Global state
_is_configured = False
_loggers = {}


def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    force: bool = False
) -> None:
    """Setup logging from configuration.
    
    Args:
        config: Configuration dictionary (can be full TaskDialogue config)
        force: Force reconfiguration even if already configured
        
    Example:
        >>> from taskdialogue.utils.config import load_config
        >>> config = load_config("configs/default.yaml")
        >>> setup_logging(config)
    """
    global _is_configured
    
    if _is_configured and not force:
        return
    
    # Remove default handler
    logger.remove()
    
    # Extract logging config
    if config is None:
        log_config = DEFAULT_LOG_CONFIG
    else:
        # Support both config.get() (Config object) and dict access
        if hasattr(config, 'get'):
            log_config = {
                "level": config.get("logging.level", DEFAULT_LOG_CONFIG["level"]),
                "format": config.get("logging.format", DEFAULT_LOG_CONFIG["format"]),
                "colorize": config.get("logging.colorize", DEFAULT_LOG_CONFIG["colorize"]),
                "backtrace": config.get("logging.backtrace", DEFAULT_LOG_CONFIG["backtrace"]),
                "diagnose": config.get("logging.diagnose", DEFAULT_LOG_CONFIG["diagnose"]),
                "file": config.get("logging.file"),
                "rotation": config.get("logging.rotation", "500 MB"),
                "retention": config.get("logging.retention", "10 days"),
            }
        else:
            # Plain dict
            log_config = config.get("logging", DEFAULT_LOG_CONFIG)
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=log_config.get("format", DEFAULT_LOG_CONFIG["format"]),
        level=log_config.get("level", DEFAULT_LOG_CONFIG["level"]),
        colorize=log_config.get("colorize", DEFAULT_LOG_CONFIG["colorize"]),
        backtrace=log_config.get("backtrace", DEFAULT_LOG_CONFIG["backtrace"]),
        diagnose=log_config.get("diagnose", DEFAULT_LOG_CONFIG["diagnose"]),
    )
    
    # Add file handler if specified
    log_file = log_config.get("file")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_path),
            format=log_config.get("format", DEFAULT_LOG_CONFIG["format"]),
            level=log_config.get("level", DEFAULT_LOG_CONFIG["level"]),
            rotation=log_config.get("rotation", "500 MB"),
            retention=log_config.get("retention", "10 days"),
            compression="zip",
            backtrace=log_config.get("backtrace", DEFAULT_LOG_CONFIG["backtrace"]),
            diagnose=log_config.get("diagnose", DEFAULT_LOG_CONFIG["diagnose"]),
        )
        logger.info(f"Logging to file: {log_path}")
    
    _is_configured = True
    logger.info(f"Logging configured with level: {log_config.get('level')}")


def get_logger(name: str) -> Any:
    """Get a logger instance for a module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Hello world")
    """
    global _is_configured
    
    # Auto-configure with defaults if not yet configured
    if not _is_configured:
        setup_logging()
    
    # Return loguru logger with context
    return logger.bind(name=name)


def set_level(level: str) -> None:
    """Change the logging level at runtime.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Example:
        >>> set_level("DEBUG")
    """
    # This will affect all handlers
    logger.level(level)
    logger.info(f"Logging level changed to: {level}")


def disable_module(module_name: str) -> None:
    """Disable logging for a specific module.
    
    Args:
        module_name: Module name to disable
        
    Example:
        >>> disable_module("httpx")
    """
    logger.disable(module_name)


def enable_module(module_name: str) -> None:
    """Enable logging for a specific module.
    
    Args:
        module_name: Module name to enable
    """
    logger.enable(module_name)


# Convenience function for quick setup
def quick_setup(level: str = "INFO", colorize: bool = True) -> None:
    """Quick setup for simple use cases.
    
    Args:
        level: Log level
        colorize: Whether to use colors
        
    Example:
        >>> quick_setup("DEBUG")
    """
    config = {
        "level": level,
        "colorize": colorize,
    }
    setup_logging({"logging": config}, force=True)


# Auto-configure with defaults on import (can be overridden)
if not _is_configured:
    # Silent initial setup - just add basic handler
    logger.remove()
    logger.add(
        sys.stderr,
        format=DEFAULT_LOG_CONFIG["format"],
        level="INFO",
        colorize=True,
    )
    _is_configured = True

