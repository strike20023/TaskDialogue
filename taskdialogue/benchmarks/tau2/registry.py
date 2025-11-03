"""Local registry for domains, tasks, agents, users, and tools (skeleton)."""

from typing import Any, Callable, Dict, List


class LocalRegistry:
    def __init__(self) -> None:
        self._users: Dict[str, Any] = {}
        self._agents: Dict[str, Any] = {}
        self._domains: Dict[str, Callable[[], Any]] = {}
        self._tasks: Dict[str, Callable[[], List[Any]]] = {}

    def register_user(self, name: str, user: Any) -> None:
        self._users[name] = user

    def register_agent(self, name: str, agent: Any) -> None:
        self._agents[name] = agent

    def register_domain(self, name: str, get_environment: Callable[[], Any]) -> None:
        self._domains[name] = get_environment

    def register_tasks(self, name: str, get_tasks: Callable[[], List[Any]]) -> None:
        self._tasks[name] = get_tasks

    def get_user(self, name: str) -> Any:
        return self._users[name]

    def get_agent(self, name: str) -> Any:
        return self._agents[name]

    def get_domain(self, name: str) -> Callable[[], Any]:
        return self._domains[name]
    
    def get_env_constructor(self, name: str) -> Callable[[], Any]:
        """Get environment constructor (alias for get_domain for compatibility)"""
        return self.get_domain(name)

    def get_tasks(self, name: str) -> Callable[[], List[Any]]:
        return self._tasks[name]


registry = LocalRegistry()

# Pre-register domains with local loaders
try:
    # Airline
    from taskdialogue.benchmarks.tau2.domains.airline.environment import (
        get_environment as airline_get_environment,
        get_tasks as airline_get_tasks,
    )
    registry.register_domain("airline", airline_get_environment)
    registry.register_tasks("airline", airline_get_tasks)
    
    # Retail
    from taskdialogue.benchmarks.tau2.domains.retail.environment import (
        get_environment as retail_get_environment,
        get_tasks as retail_get_tasks,
    )
    registry.register_domain("retail", retail_get_environment)
    registry.register_tasks("retail", retail_get_tasks)
    
    # Telecom
    from taskdialogue.benchmarks.tau2.domains.telecom.environment import (
        get_environment_manual_policy as telecom_get_environment,
        get_tasks as telecom_get_tasks,
    )
    registry.register_domain("telecom", telecom_get_environment)
    registry.register_tasks("telecom", telecom_get_tasks)
    
except Exception as e:
    # Log error but don't fail module import
    import warnings
    warnings.warn(f"Failed to register domains: {e}")


