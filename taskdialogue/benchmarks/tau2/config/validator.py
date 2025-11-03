"""Configuration validation for Tau2-bench."""
from taskdialogue.core.utils.logger import get_logger


logger = get_logger(__name__)

from typing import Any, Dict, List, Optional


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_tau2_config(config: Dict[str, Any]) -> List[str]:
    """Validate Tau2-bench configuration.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of warning messages (empty if all valid)
        
    Raises:
        ConfigValidationError: If critical validation fails
    """
    warnings = []
    
    # Check benchmark type
    benchmark_type = config.get("benchmark.type")
    if benchmark_type and benchmark_type != "tau2":
        warnings.append(f"Benchmark type is '{benchmark_type}', expected 'tau2'")
    
    # Check domain
    domain = config.get("tau2.domain")
    if not domain:
        raise ConfigValidationError("tau2.domain is required")
    
    valid_domains = ["airline", "retail", "telecom"]
    if domain not in valid_domains:
        warnings.append(
            f"Domain '{domain}' may not be supported. "
            f"Valid domains: {', '.join(valid_domains)}"
        )
    
    # Check num_tasks
    num_tasks = config.get("tau2.trials.num_tasks")
    if num_tasks is not None and num_tasks <= 0:
        raise ConfigValidationError(f"num_tasks must be positive, got {num_tasks}")
    
    # Check num_trials
    num_trials = config.get("tau2.trials.num_trials", 1)
    if num_trials <= 0:
        raise ConfigValidationError(f"num_trials must be positive, got {num_trials}")
    
    # Check limits
    max_steps = config.get("tau2.limits.max_steps", 200)
    if max_steps <= 0:
        raise ConfigValidationError(f"max_steps must be positive, got {max_steps}")
    
    max_errors = config.get("tau2.limits.max_errors", 10)
    if max_errors < 0:
        raise ConfigValidationError(f"max_errors must be non-negative, got {max_errors}")
    
    # Check model configuration
    provider = config.get("model.agent.provider")
    if not provider:
        warnings.append("model.agent.provider not specified")
    
    model_name = config.get("model.agent.name")
    if not provider:
        warnings.append("model.agent.name not specified")
    
    # Check tools profile
    tools_profile = config.get("tau2.tools.profile", "original")
    valid_profiles = ["original", "sql"]
    if tools_profile not in valid_profiles:
        warnings.append(
            f"Tools profile '{tools_profile}' may not be supported. "
            f"Valid profiles: {', '.join(valid_profiles)}"
        )
    
    return warnings


def validate_and_warn(config: Dict[str, Any]) -> None:
    """Validate configuration and print warnings.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ConfigValidationError: If critical validation fails
    """
    warnings = validate_tau2_config(config)
    
    if warnings:
        logger.info("\n⚠️  Configuration Warnings:")
        for warning in warnings:
            logger.info(f"  - {warning}")
        logger.info()

