"""TaskDialogue adapters for Tau2-bench integration.

This package contains all TaskDialogue-specific adapters that bridge
TaskDialogue's agents and tools with the Tau2-bench framework.

Submodules:
- agent: Agent adapters (conversion utils)
- tools: Tool adapters and registry
"""

from taskdialogue.benchmarks.tau2.adapters.agent.conversion import (
    convert_tau2_to_taskdialogue_predictions,
)

__all__ = [
    "convert_tau2_to_taskdialogue_predictions",
]

