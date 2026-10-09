"""A scripted chat model that answers provider-native structured output the way LangChain's ``include_raw`` does."""

from typing import Any, ClassVar

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.runnables import RunnableLambda


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    )


class StructuredFakeModel(FakeMessagesListChatModel):
    """Return ``{"raw", "parsed", "parsing_error"}``; parsing mirrors a JSON-schema parser over the raw text."""

    seen_messages: ClassVar[list[list[Any]]] = []
    structured: ClassVar[list[tuple[type, dict[str, object]]]] = []

    def bind_tools(self, *_args: Any, **_kwargs: Any):
        raise AssertionError("structured decisions must not bind tools")

    def _generate(self, messages: list[Any], *args: Any, **kwargs: Any):
        type(self).seen_messages.append(list(messages))
        return super()._generate(messages, *args, **kwargs)

    def with_structured_output(self, schema, **options):
        type(self).structured.append((schema, dict(options)))

        def run(messages):
            raw = self.invoke(messages)
            try:
                return {"raw": raw, "parsed": schema.model_validate_json(_text(raw.content)), "parsing_error": None}
            except ValueError as exc:
                return {"raw": raw, "parsed": None, "parsing_error": exc}

        return RunnableLambda(run)
