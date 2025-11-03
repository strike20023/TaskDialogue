"""Prompt templates for Tau2-bench agents and user simulators.

This module centralizes all prompt management for the Tau2-bench framework,
following the same pattern as taskdialogue.prompts.

Organization:
- prompts.py: All prompt templates and constants
- Getter functions for clean API access
"""

from taskdialogue.benchmarks.tau2.prompts.prompts import (
    # Agent prompts
    get_agent_system_prompt,
    get_agent_system_prompt_sql,
    get_agent_instruction,
    AGENT_INSTRUCTION,
    # User simulator prompts
    get_user_simulator_system_prompt,
    get_user_simulator_guidelines,
    USER_SIMULATOR_GUIDELINES,
    USER_SIMULATOR_GUIDELINES_TOOLS,
    # Utilities
    validate_prompts,
    list_available_prompts,
)

from taskdialogue.benchmarks.tau2.prompts.policies import (
    # Domain policies
    get_domain_policy,
    list_available_domains,
    get_policy_stats,
    AIRLINE_POLICY,
    RETAIL_POLICY,
)

__all__ = [
    # Agent prompts
    "get_agent_system_prompt",
    "get_agent_system_prompt_sql",
    "get_agent_instruction",
    "AGENT_INSTRUCTION",
    # User simulator prompts
    "get_user_simulator_system_prompt",
    "get_user_simulator_guidelines",
    "USER_SIMULATOR_GUIDELINES",
    "USER_SIMULATOR_GUIDELINES_TOOLS",
    # Domain policies
    "get_domain_policy",
    "list_available_domains",
    "get_policy_stats",
    "AIRLINE_POLICY",
    "RETAIL_POLICY",
    # Utilities
    "validate_prompts",
    "list_available_prompts",
]

