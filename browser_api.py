"""The Browser operations' request objects (ADR-0117): a step, a fold, a final answer, and the completeness check."""

from typing import Any, Literal

import browser_step
from api_input import ClosedInput, ProviderInput
from protocol.team.http.v1 import browser as browser_protocol
from pydantic import Field


class BrowserStepInput(ClosedInput):
    """One Browser agent call (ADR-0117): the Team's provider, the task, its thread, and the bounded current page."""

    provider: ProviderInput
    objective: str = Field(min_length=1, max_length=browser_protocol.MAX_OBJECTIVE_CHARS)
    language: Literal["en", "pt-BR"]
    # The task's thread, admitted whole by the Browser protocol before any model call.
    thread: dict[str, Any]
    observation: str = Field(max_length=browser_protocol.MAX_OBSERVATION_BYTES)
    screenshot: str | None = Field(max_length=browser_step.MAX_SCREENSHOT_BYTES * 4 // 3 + 4)
    # The completed downloads present in the task's Browser so far, or null when Team could not read them.
    downloads: dict[str, Any] | None

    def runtime_request(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "language": self.language,
            "thread": self.thread,
            "observation": self.observation,
            "screenshot": self.screenshot,
            "downloads": self.downloads,
        }


class BrowserFoldInput(BrowserStepInput):
    """A fold: the step request it extends, and the step numbers whose pages are about to leave the thread."""

    first: int = Field(ge=1, le=browser_protocol.MAX_STEPS)
    last: int = Field(ge=1, le=browser_protocol.MAX_STEPS)

    def runtime_request(self) -> dict[str, object]:
        return {**super().runtime_request(), "first": self.first, "last": self.last}


class BrowserCheckInput(ClosedInput):
    """The completeness check: the Team's provider, the objective, and the agent's done answer."""

    provider: ProviderInput
    objective: str = Field(min_length=1, max_length=browser_protocol.MAX_OBJECTIVE_CHARS)
    language: Literal["en", "pt-BR"]
    summary: str = Field(min_length=1, max_length=browser_protocol.MAX_SUMMARY_BYTES)

    def runtime_request(self) -> dict[str, object]:
        return {"objective": self.objective, "language": self.language, "summary": self.summary}
