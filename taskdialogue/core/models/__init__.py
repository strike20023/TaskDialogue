"""Model abstraction layer for different inference backends."""

from taskdialogue.core.models.factory import create_model, create_model_from_config
from taskdialogue.core.models.base import BaseModel, ModelResponse

__all__ = ["create_model", "create_model_from_config", "BaseModel", "ModelResponse"]

