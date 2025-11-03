"""Configuration management for Tau2-bench integration.

This module provides configuration management, validation, and defaults
for Tau2-bench runs within TaskDialogue.
"""

from taskdialogue.benchmarks.tau2.config.defaults import *
from taskdialogue.benchmarks.tau2.config.manager import ConfigManager, RunConfig
from taskdialogue.benchmarks.tau2.config.validator import validate_tau2_config, validate_and_warn

__all__ = [
    # Defaults
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MAX_ERRORS",
    "DEFAULT_SEED",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_NUM_TRIALS",
    # Manager
    "ConfigManager",
    "RunConfig",
    # Validator
    "validate_tau2_config",
    "validate_and_warn",
]

