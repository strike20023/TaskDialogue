"""Configuration-driven CLI for running Tau2-bench inside TaskDialogue.

This module provides the main entry points for running Tau2-bench from
configuration files or programmatically.
"""

from typing import Any, Dict

from taskdialogue.benchmarks.tau2.config import RunConfig, ConfigManager, validate_and_warn
from taskdialogue.benchmarks.tau2.run import run_domain


def run_from_config(config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Run Tau2-bench from TaskDialogue configuration.
    
    Args:
        config: Complete TaskDialogue configuration (including model, api, tau2 sections)
        verbose: Whether to print configuration summary
        
    Returns:
        Dictionary containing simulations and metrics
        
    Example:
        >>> from taskdialogue.core.utils.config import load_config
        >>> config = load_config("configs/tau2_default.yaml")
        >>> results = run_from_config(config)
    """
    # Validate configuration
    validate_and_warn(config)
    
    # Create run configuration from TaskDialogue config
    run_cfg = RunConfig.from_config(config)
    
    # Print summary if verbose
    if verbose:
        ConfigManager.print_config_summary(run_cfg)
    
    # Run the benchmark
    return run_domain(run_cfg, taskdialogue_config=config)


