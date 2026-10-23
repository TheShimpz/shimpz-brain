"""The closed request objects every Brain endpoint builds its inputs from, and the provider a request names."""

from typing import Literal

import agent_runtime
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class ClosedInput(BaseModel):
    """A request object refused whole, with one constant error, when it carries any field it does not declare.

    Pydantic's own `extra="forbid"` reports each unknown field separately, and every error costs about a kilobyte of
    memory, far more than the few body bytes that create it. One error per object keeps the errors a body can cause
    bounded by the objects its lists admit.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def refuse_unknown_fields(cls, value: object) -> object:
        if isinstance(value, dict) and any(key not in cls.model_fields for key in value):
            raise ValueError("unknown field")
        return value


class ProviderInput(ClosedInput):
    provider: Literal["anthropic", "openai"]
    model: str = Field(min_length=1, max_length=128)
    api_key: SecretStr = Field(min_length=agent_runtime.MIN_API_KEY_CHARS, max_length=agent_runtime.MAX_API_KEY_CHARS)

    def runtime(self) -> agent_runtime.ProviderConfig:
        # Team names the Team's chat model; a model the Brain only runs itself is never one.
        if self.model not in agent_runtime.CHAT_MODELS_BY_PROVIDER.get(self.provider, frozenset()):
            raise agent_runtime.RuntimeContractError("unsupported model for provider")
        return agent_runtime.ProviderConfig(self.provider, self.model, self.api_key.get_secret_value())
