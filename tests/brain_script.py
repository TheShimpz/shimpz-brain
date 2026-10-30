"""A scripted stand-in for the Brain's HTTP client, shared by the provider-free and binary-free harness tests.

Each scripted answer is what one Brain operation answers: a list of ``(assistant_id, action, input)`` requests (an
``action-required`` turn), a string (a completed turn with that reply), or a ``brain_api.BrainError`` to raise. It
says nothing about how a real model or the real Brain behaves; it proves only the harness's own scoring.
"""

from collections.abc import Mapping

from eval import brain_api
from eval import cost as eval_cost

PROVIDER = brain_api.Provider("openai", "gpt-6-luna", "secret-test-key")


def usage(**counts: int) -> dict[str, int]:
    """One operation's usage counts: every field, zero unless given."""
    return dict.fromkeys(eval_cost.FIELDS, 0) | counts


ONE_CALL = usage(model_calls=1, input_tokens=1000, output_tokens=10, provider_requests=1)


def answer(scripted: object, number: int, counts: Mapping[str, int] = ONE_CALL) -> brain_api.Turn:
    """The turn one scripted answer stands for; its requests get interrupt ids unique to the script."""
    if isinstance(scripted, str):
        return brain_api.Turn("completed", scripted, (), dict(counts))
    requests = tuple(
        brain_api.ActionRequest(f"i{number}-{index}", assistant, action, dict(arguments))
        for index, (assistant, action, arguments) in enumerate(scripted)
    )
    return brain_api.Turn("action-required", "", requests, dict(counts))


class ScriptedBrain:
    """Answers each start or resume with the next scripted answer and records what the harness sent."""

    def __init__(self, *answers: object, counts: Mapping[str, int] = ONE_CALL) -> None:
        self.answers = list(answers)
        self.counts = counts
        self.calls: list[tuple[str, brain_api.TurnContext, object]] = []

    def _next(self) -> brain_api.Turn:
        scripted = self.answers.pop(0)
        if isinstance(scripted, brain_api.BrainError):
            raise scripted
        return answer(scripted, len(self.calls) - 1, self.counts)

    def start(self, context: brain_api.TurnContext, message: str, **options: object) -> brain_api.Turn:
        self.calls.append(("start", context, {"message": message, **options}))
        return self._next()

    def resume(self, context: brain_api.TurnContext, results: Mapping[str, object]) -> brain_api.Turn:
        self.calls.append(("resume", context, dict(results)))
        return self._next()
