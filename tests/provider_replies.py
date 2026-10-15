"""Provider configurations and wire reply bodies shared by provider-free Brain tests.

Each body is what a provider's HTTP endpoint answers for one completed text reply; tests serve it through a mock
transport and override the fields a case is about.
"""

import agent_runtime

MODELS = {"openai": "gpt-6.1-sol", "anthropic": "claude-sonnet-5-5"}


def provider(name: str = "openai") -> agent_runtime.ProviderConfig:
    """A valid configuration of the provider's default test model, with a low-entropy fake key."""
    return agent_runtime.ProviderConfig(name, MODELS[name], "secret-test-key")


def openai_reply(
    text: str = "Done.", *, model: str = MODELS["openai"], usage: dict[str, object] | None = None
) -> dict[str, object]:
    """One completed OpenAI Responses body holding a single text message."""
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": usage or {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def anthropic_reply(text: str = "Done.", *, usage: dict[str, object] | None = None) -> dict[str, object]:
    """One finished Anthropic Messages body holding a single text block."""
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": MODELS["anthropic"],
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 10, "output_tokens": 2},
    }
