"""Direct model-provider clients for the Brain runtime: one credential holder per call over a cancellable pool.

The API key is held only by the short-lived client a call builds; it never enters graph state, and the connection pool
the clients share carries no credential.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import action_tool
import httpx
import provider_cancel
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr
from runtime_errors import RuntimeContractError

if TYPE_CHECKING:
    from agent_runtime import ProviderConfig

DECISION_TIMEOUT_SECONDS = 10.0
# One Routine compile may write a plan of hundreds of steps (ADR-0092 amendment, 2026-10-05, scale); Team waits for its
# turn up to 300 seconds, so the provider is given 240.
COMPILE_TIMEOUT_SECONDS = 240.0
# One retry recovers a rare stalled or failed structured route call; the call is stateless and tool-free (ADR-0071).
DECISION_MAX_RETRIES = 1


def provider_model(
    config: ProviderConfig,
    *,
    http_client: httpx.Client | None = None,
    decision: bool = False,
    retries: int | None = None,
    timeout: float | None = None,
) -> BaseChatModel:
    """Create one direct provider client; the API key is never put in graph state."""
    secret = SecretStr(config.api_key)
    common = {
        "model": config.model,
        "api_key": secret,
        "timeout": timeout or (DECISION_TIMEOUT_SECONDS if decision else 60.0),
        "max_retries": retries if retries is not None else DECISION_MAX_RETRIES if decision else 2,
    }
    if config.provider == "openai":
        # The owner keeps no conversation on OpenAI's side: every Responses call is unstored, and reasoning is carried
        # between tool calls as encrypted content the adapter replays.
        openai = {**common, "use_responses_api": True, "store": False, "include": ["reasoning.encrypted_content"]}
        if decision:
            openai["reasoning_effort"] = "low"
        elif config.effort is not None:
            openai["reasoning_effort"] = config.effort
        if http_client is not None:
            openai["http_client"] = http_client
        return _action_chat_openai()(**openai)
    if config.provider == "anthropic":
        effort = "low" if decision else config.effort
        anthropic = {**common, **({"effort": effort} if effort is not None else {})}
        model = _pooled_chat_anthropic()(**anthropic)
        model._http_client = http_client
        return model
    raise RuntimeContractError("unsupported model provider")


@functools.cache
def _action_chat_openai() -> type[BaseChatModel]:
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_openai import ChatOpenAI

    class ActionChatOpenAI(ChatOpenAI):
        """ChatOpenAI that sends every Action tool with ``strict: false`` and its declared ``required`` list.

        The Responses API treats a function tool sent without ``strict`` as strict when it can and then requires every
        property, so the model invents a value for each optional Action property (ADR-0094). Optional stays optional;
        the Action tool still validates every payload and Team validates again. Brain's own tools keep the provider
        default.
        """

        def bind_tools(self, tools, **kwargs):
            return super().bind_tools(
                [
                    convert_to_openai_tool(tool, strict=False) if isinstance(tool, action_tool.ActionTool) else tool
                    for tool in tools
                ],
                **kwargs,
            )

    return ActionChatOpenAI


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

    def single_attempt(self, config: ProviderConfig, *, decision: bool = False) -> BaseChatModel:
        """A chat model whose SDK client never retries a call by itself.

        An attachment turn retries explicitly, reserving its budget per attempt (ADR-0093); a Routine compile or
        recovery decision is exactly one billed call (ADR-0092). The retry count is fixed when the SDK client is built,
        so a copy of a retrying model cannot become a single-attempt one.
        """
        return provider_model(config, http_client=self._http_client, decision=decision, retries=0)

    def compile(self, config: ProviderConfig) -> BaseChatModel:
        """A single-attempt model for one Routine compile, given the longer time a plan of hundreds of steps needs."""
        return provider_model(config, http_client=self._http_client, retries=0, timeout=COMPILE_TIMEOUT_SECONDS)

    def decision(self, config: ProviderConfig) -> BaseChatModel:
        """Build a short model with at most one retry for one structured routing decision."""
        return provider_model(config, http_client=self._http_client, decision=True)

    def close(self) -> None:
        self._http_client.close()
