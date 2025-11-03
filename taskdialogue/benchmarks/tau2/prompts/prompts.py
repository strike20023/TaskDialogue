"""
from taskdialogue.core.utils.logger import get_logger

logger = get_logger(__name__)

Prompt templates for Tau2-bench agents and user simulators.

This module contains all prompt templates used in the Tau2 benchmark framework.
Prompts are organized by component (agent, user simulator) and use a consistent
pattern of constants + getter functions.

DESIGN PRINCIPLES:
- Centralized: All prompts in one place
- Versioned: Track changes and iterations
- Documented: Clear purpose and usage
- Testable: Easy to test and iterate

CHANGE LOG:
- 2025-10-30: Initial migration - consolidated prompts from multiple files
  * Agent prompts from adapters/agent/prompts/
  * User simulator prompts from user/user_simulator.py
  * Following taskdialogue.prompts pattern
"""

from typing import Optional


# ==============================================================================
# AGENT PROMPTS
# ==============================================================================
# These prompts define the behavior of the customer service agent in Tau2-bench.
# The agent helps users according to domain-specific policies.
# ==============================================================================

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

IMPORTANT:
- When sending a message to the user, output the message text directly in natural language. DO NOT wrap it in JSON format.
- When making a tool call, use the function calling mechanism provided by the system (which uses JSON internally).
- The "generate valid JSON only" rule applies ONLY to tool calls, NOT to regular messages to users.

Try to be helpful and always follow the policy.
""".strip()

AGENT_SYSTEM_PROMPT_TEMPLATE = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()

# SQL/Hybrid 模式专用的 Agent Instruction（包含 SQL 使用指南）
AGENT_INSTRUCTION_SQL = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.

You have access to SQL database query tools (for reading data):
- db_query: Execute SELECT queries to read data from the database
- db_list_tables: List all available tables in the database
- db_get_schema: Get schema information (column names, types) for a specific table

IMPORTANT: For write operations (creating, updating, deleting records), use the domain-specific business logic tools (e.g., book_reservation, cancel_reservation) instead of SQL. These tools ensure correct business logic and data consistency.

SQL BEST PRACTICES (for queries only):
1. Use db_list_tables first to discover available tables
2. Use db_get_schema to understand table structure before writing queries
3. Write clear SELECT queries with appropriate WHERE clauses
4. Always validate query results before making decisions
5. For complex queries, break them into smaller steps if needed
6. Query results are automatically formatted and limited for readability

TOOL USAGE STRATEGY:
- Use SQL tools (db_query, db_list_tables, db_get_schema) for reading/searching data
- Use domain-specific tools (book_*, cancel_*, update_*, etc.) for write operations
- This hybrid approach gives you flexible queries while maintaining data integrity

IMPORTANT:
- When sending a message to the user, output the message text directly in natural language. DO NOT wrap it in JSON format.
- When making a tool call, use the function calling mechanism provided by the system (which uses JSON internally).
- The "generate valid JSON only" rule applies ONLY to tool calls, NOT to regular messages to users.

Try to be helpful and always follow the policy.
""".strip()

AGENT_SYSTEM_PROMPT_SQL_TEMPLATE = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<database_info>
{schema_info}
</database_info>
""".strip()


def get_agent_instruction() -> str:
    """Get the agent instruction text.
    
    Returns:
        Agent instruction string
    """
    return AGENT_INSTRUCTION


def get_agent_system_prompt(
    domain_policy: str,
    agent_instruction: Optional[str] = None
) -> str:
    """Generate system prompt for agent with domain policy (original mode).
    
    Args:
        domain_policy: Domain-specific policy text (e.g., airline policy)
        agent_instruction: Override default agent instruction (optional)
        
    Returns:
        Formatted system prompt for agent
        
    Example:
        >>> policy = load_policy("airline")
        >>> prompt = get_agent_system_prompt(policy)
    """
    if agent_instruction is None:
        agent_instruction = AGENT_INSTRUCTION
    
    return AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        agent_instruction=agent_instruction,
        domain_policy=domain_policy,
    )


def get_agent_system_prompt_sql(
    domain_policy: str,
    schema_info: Optional[str] = None,
    agent_instruction: Optional[str] = None
) -> str:
    """Generate system prompt for agent in SQL mode (with SQL-specific instructions).
    
    Args:
        domain_policy: Domain-specific policy text (e.g., airline policy)
        schema_info: Database schema information (optional, will be fetched if not provided)
        agent_instruction: Override default SQL agent instruction (optional)
        
    Returns:
        Formatted system prompt for agent in SQL mode
        
    Example:
        >>> policy = load_policy("airline")
        >>> schema = "Tables: flights, bookings, users..."
        >>> prompt = get_agent_system_prompt_sql(policy, schema)
    """
    if agent_instruction is None:
        agent_instruction = AGENT_INSTRUCTION_SQL
    
    # 如果没有提供 schema_info，添加提示信息
    if schema_info is None:
        schema_info = "Use db_list_tables and db_get_schema tools to discover database structure."
    
    return AGENT_SYSTEM_PROMPT_SQL_TEMPLATE.format(
        agent_instruction=agent_instruction,
        domain_policy=domain_policy,
        schema_info=schema_info,
    )


# ==============================================================================
# USER SIMULATOR PROMPTS
# ==============================================================================
# These prompts define the behavior of the user simulator in Tau2-bench.
# The simulator acts as a user to interact with the agent for evaluation.
# 
# All guidelines are embedded as constants - no external .md files needed.
# ==============================================================================

# User simulator guidelines (without tools)
USER_SIMULATOR_GUIDELINES = """# User Simulation Guidelines
You are playing the role of a customer contacting a customer service representative. 
Your goal is to simulate realistic customer interactions while following specific scenario instructions.

## Core Principles
- Generate one message at a time, maintaining natural conversation flow.
- ALWAYS respond in natural, conversational language. NEVER output JSON format, structured data, or code blocks. Your messages should sound like a real customer speaking.
- Strictly follow the scenario instructions you have received.
- Never make up or hallucinate information not provided in the scenario instructions. Information that is not provided in the scenario instructions should be considered unknown or unavailable.
- Avoid repeating the exact instructions verbatim. Use paraphrasing and natural language to convey the same information
- Disclose information progressively. Wait for the agent to ask for specific information before providing it.

## Task Completion
- The goal is to continue the conversation until the task is complete.
- If the instruction goal is satisified, generate the '###STOP###' token to end the conversation.
- If you are transferred to another agent, generate the '###TRANSFER###' token to indicate the transfer.
- If you find yourself in a situation in which the scenario does not provide enough information for you to continue the conversation, generate the '###OUT-OF-SCOPE###' token to end the conversation.

Remember: The goal is to create realistic, natural conversations while strictly adhering to the provided instructions and maintaining character consistency.""".strip()

# User simulator guidelines (with tools enabled)
USER_SIMULATOR_GUIDELINES_TOOLS = """# User Simulation Guidelines

You are playing the role of a customer contacting a customer service representative agent. 
Your goal is to simulate realistic customer interactions while following specific scenario instructions.
You have some tools to perform the actions on your end that might be requested by the agent to diagnose and resolve your issue.

## Core Principles
- Generate one message at a time, maintaining natural conversation flow.
- ALWAYS respond in natural, conversational language. NEVER output JSON format, structured data, API response formats (like {"status": "success", "data": {...}}), or code blocks. Your messages should sound like a real customer speaking.
- CRITICAL: When the agent asks for information, respond in natural language, not in structured formats. For example, if asked about your reservation, say "My reservation is for flight AA2471 on July 15th" NOT {"status": "success", "data": {"flight": "AA2471"}}.
- At each turn you can either:
    - Send a message to the agent.
    - Make a tool call to perform an action requested by the agent.
    - You cannot do both at the same time.
- Strictly follow the scenario instructions you have received.
- Never make up or hallucinate information not provided in the scenario instructions. Information that is not provided in the scenario instructions should be considered unknown or unavailable.
- Never make up the results of tool calls that the agent has requested, you must ground your responses based on the results of tool calls if the agent has requested.
- If you made an error in a tool call and get an error message, fix the error and try again.
- All the information you provide to the agent must be grounded in the information provided in the scenario instructions or the results of tool calls.
- Avoid repeating the exact instructions verbatim. Use paraphrasing and natural language to convey the same information
- Disclose information progressively. Wait for the agent to ask for specific information before providing it.
- Only call a tool if the agent has requested it or if it is necessary to answer a question the agent has asked. Ask clarifying questions if you do not know what action to take.
- If the agent asks multiple actions to perform, state that you cannot perform multiple actions at once, and ask the agent to instruct you one action at a time.
- Your messages when performing tool calls will not be displayed to the agent, only the messages without tool calls will be displayed to the agent.

## Task Completion
- The goal is to continue the conversation until the task is complete.
- If the instruction goal is satisified, generate the '###STOP###' token to end the conversation.
- If you have been transferred to another agent, generate the '###TRANSFER###' token to indicate the transfer. Only do this after the agent has clearly indicated that you are being transferred.
- If you find yourself in a situation in which the scenario does not provide enough information for you to continue the conversation, generate the '###OUT-OF-SCOPE###' token to end the conversation.

Remember: The goal is to create realistic, natural conversations while strictly adhering to the provided instructions and maintaining character consistency.""".strip()

USER_SIMULATOR_SYSTEM_PROMPT_TEMPLATE = """
{global_user_sim_guidelines}

<scenario>
{instructions}
</scenario>
""".strip()


def get_user_simulator_guidelines(use_tools: bool = False) -> str:
    """Get global user simulator guidelines.
    
    Args:
        use_tools: Whether to use the tools-enabled guidelines
        
    Returns:
        User simulator guidelines text (embedded as constant, no file I/O)
    """
    if use_tools:
        return USER_SIMULATOR_GUIDELINES_TOOLS
    else:
        return USER_SIMULATOR_GUIDELINES


def get_user_simulator_system_prompt(
    instructions: str,
    use_tools: bool = False
) -> str:
    """Generate system prompt for user simulator.
    
    Args:
        instructions: Task-specific user instructions (scenario)
        use_tools: Whether user simulator has access to tools
        
    Returns:
        Formatted system prompt for user simulator
        
    Example:
        >>> instructions = task.user_instructions
        >>> prompt = get_user_simulator_system_prompt(instructions)
    """
    guidelines = get_user_simulator_guidelines(use_tools=use_tools)
    
    return USER_SIMULATOR_SYSTEM_PROMPT_TEMPLATE.format(
        global_user_sim_guidelines=guidelines,
        instructions=instructions,
    )


# ==============================================================================
# EVALUATION PROMPTS (Future)
# ==============================================================================
# Placeholder for evaluation-related prompts if needed
# ==============================================================================

# TODO: Add evaluation prompts if needed
# - NL assertions prompt
# - Environment interface prompt
# - etc.


# ==============================================================================
# UTILITIES
# ==============================================================================

def list_available_prompts() -> dict:
    """List all available prompts in this module.
    
    Returns:
        Dictionary mapping prompt names to descriptions
    """
    return {
        "agent": {
            "instruction": "Agent instruction text",
            "system_prompt": "Agent system prompt with policy (requires domain_policy)",
        },
        "user_simulator": {
            "guidelines": "User simulator guidelines from file",
            "system_prompt": "User simulator system prompt (requires instructions)",
        },
    }


def validate_prompts() -> bool:
    """Validate that all required prompts are available.
    
    Returns:
        True if all prompts are defined
    """
    try:
        # Check that all prompt constants are defined
        assert AGENT_INSTRUCTION, "AGENT_INSTRUCTION is empty"
        assert USER_SIMULATOR_GUIDELINES, "USER_SIMULATOR_GUIDELINES is empty"
        assert USER_SIMULATOR_GUIDELINES_TOOLS, "USER_SIMULATOR_GUIDELINES_TOOLS is empty"
        return True
    except AssertionError as e:
        logger.info(f"⚠️  Prompt validation failed: {e}")
        return False

