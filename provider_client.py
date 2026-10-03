"""Direct model-provider clients for the Brain runtime: one credential holder per call over a cancellable pool.

The API key is held only by the short-lived client a call builds; it never enters graph state, and the connection pool
the clients share carries no credential.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import httpx
import provider_cancel
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr
from runtime_errors import RuntimeContractError

if TYPE_CHECKING:
    from agent_runtime import ProviderConfig

DECISION_TIMEOUT_SECONDS = 10.0
# One retry recovers a rare stalled or failed structured route call; the call is stateless and tool-free (ADR-0071).
DECISION_MAX_RETRIES = 1


def provider_model(
    config: ProviderConfig,
    *,
    http_client: httpx.Client | None = None,
    decision: bool = False,
    retries: int | None = None,
) -> BaseChatModel:
    """Create one direct provider client; the API key is never put in graph state."""
    secret = SecretStr(config.api_key)
    common = {
        "model": config.model,
        "api_key": secret,
        "timeout": DECISION_TIMEOUT_SECONDS if decision else 60.0,
        "max_retries": retries if retries is not None else DECISION_MAX_RETRIES if decision else 2,
    }
    if config.provider == "openai":
        from langchain_openai import ChatOpenAI

        # The owner keeps no conversation on OpenAI's side: every Responses call is unstored, and reasoning is carried
        # between tool calls as encrypted content the adapter replays.
        openai = {**common, "use_responses_api": True, "store": False, "include": ["reasoning.encrypted_content"]}
        if decision:
            openai["reasoning_effort"] = "low"
        elif config.effort is not None:
            openai["reasoning_effort"] = config.effort
        if http_client is not None:
            openai["http_client"] = http_client
        return ChatOpenAI(**openai)
    if config.provider == "anthropic":
        effort = "low" if decision else config.effort
        anthropic = {**common, **({"effort": effort} if effort is not None else {})}
        model = _pooled_chat_anthropic()(**anthropic)
        model._http_client = http_client
        return model
    raise RuntimeContractError("unsupported model provider")


@functools.cache
def _pooled_chat_anthropic() -> type[BaseChatModel]:
    import anthropic
    from langchain_anthropic import ChatAnthropic
    from pydantic import PrivateAttr

    class PooledChatAnthropic(ChatAnthropic):
        """ChatAnthropic over the runtime's cancellable pool instead of langchain-anthropic's process-wide client.

        Overrides the ``_client`` of the pinned langchain-anthropic 1.4.8; only the synchronous client is used.
        """

        _http_client: httpx.Client | None = PrivateAttr(default=None)

        @functools.cached_property
        def _client(self) -> anthropic.Client:
            return anthropic.Client(**self._client_params, http_client=self._http_client)

    return PooledChatAnthropic


class ProviderModelFactory:
    """Build short-lived credential holders over one credential-free, turn-cancellable connection pool."""

    def __init__(self) -> None:
        self._http_client = provider_cancel.client()

    def __call__(self, config: ProviderConfig) -> BaseChatModel:
        return provider_model(config, http_client=self._http_client)

    def single_attempt(self, config: ProviderConfig) -> BaseChatModel:
        """A chat model without hidden SDK retries, for a turn whose retries reserve attachment budget (ADR-0093)."""
        return provider_model(config, http_client=self._http_client, retries=0)

    def decision(self, config: ProviderConfig) -> BaseChatModel:
        """Build a short model with at most one retry for one structured routing decision."""
        return provider_model(config, http_client=self._http_client, decision=True)

    def close(self) -> None:
        self._http_client.close()
