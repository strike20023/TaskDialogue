"""TaskDialogue: Unified Task-Oriented Dialogue Benchmark Framework

This package provides a unified framework for multiple dialogue benchmarks:
- MultiWOZ: Multi-domain task-oriented dialogue
- Tau2: Customer service dialogue evaluation

Architecture:
- core/: Shared abstractions and components
- benchmarks/: Benchmark-specific implementations
- cli/: Unified command-line interface

Quick Start:
    # Using CLI
    python -m taskdialogue multiwoz -c configs/multiwoz/default.yaml -n 10
    
    # Using Python API
    from taskdialogue.core.utils.config import load_config
    from taskdialogue.benchmarks.multiwoz import MultiWOZPipeline
    
    config = load_config("configs/multiwoz/default.yaml")
    pipeline = MultiWOZPipeline(config.to_dict())
    results = pipeline.run_full_pipeline()
"""

__version__ = "2.0.0"

# Setup logging on import
from taskdialogue.core.utils.logger import setup_logging
setup_logging()

__all__ = [
    "core",
    "benchmarks",
    "cli",
    "tau2_bench",
]
