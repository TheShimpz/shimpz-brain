"""Authenticated HTTP boundary for the isolated Shimpz LangGraph runtime."""

from __future__ import annotations

import hmac
import os
import sqlite3
import threading
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import agent_runtime
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

TOKEN_FILE = Path(os.environ.get("SHIMPZ_BRAIN_RUNTIME_TOKEN_FILE", "/run/shimpz-brain-runtime/token"))
STATE_PATH = Path(os.environ.get("SHIMPZ_BRAIN_RUNTIME_STATE", "/var/lib/shimpz-brain-runtime/checkpoints.sqlite3"))
MAX_TOKEN_BYTES = 4 * 1024
_PRUNE_WRITES_SQL = (
    "WITH latest AS (SELECT checkpoint_ns,MAX(checkpoint_id) AS checkpoint_id "
    "FROM checkpoints WHERE thread_id=? GROUP BY checkpoint_ns) "
    "DELETE FROM writes WHERE thread_id=? AND NOT EXISTS ("
    "SELECT 1 FROM latest WHERE latest.checkpoint_ns=writes.checkpoint_ns "
    "AND latest.checkpoint_id=writes.checkpoint_id)"
)
_PRUNE_CHECKPOINTS_SQL = (
    "WITH latest AS (SELECT checkpoint_ns,MAX(checkpoint_id) AS checkpoint_id "
    "FROM checkpoints WHERE thread_id=? GROUP BY checkpoint_ns) "
    "DELETE FROM checkpoints WHERE thread_id=? AND NOT EXISTS ("
    "SELECT 1 FROM latest WHERE latest.checkpoint_ns=checkpoints.checkpoint_ns "
    "AND latest.checkpoint_id=checkpoints.checkpoint_id)"
)


class ProviderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["anthropic", "openai"]
    model: str = Field(min_length=1, max_length=128)
    api_key: SecretStr = Field(min_length=1, max_length=16 * 1024)


class ActionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    summary: str = Field(min_length=1, max_length=2_000)
    input_schema: dict[str, Any]


class AssistantInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    genesis: str = Field(min_length=1, max_length=agent_runtime.MAX_GENESIS_BYTES)
    actions: list[ActionInput] = Field(max_length=agent_runtime.MAX_ACTIONS_PER_ASSISTANT)


class TurnContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1, max_length=256)
    team_name: str = Field(min_length=1, max_length=agent_runtime.MAX_TEAM_NAME_CHARS)
    assistants: list[AssistantInput] = Field(max_length=agent_runtime.MAX_ASSISTANTS)
    provider: ProviderInput

    @field_validator("team_name", mode="before")
    @classmethod
    def normalize_team_name(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return agent_runtime.normalize_team_name(value)

    @model_validator(mode="after")
    def bound_total_actions(self) -> Self:
        if sum(len(assistant.actions) for assistant in self.assistants) > agent_runtime.MAX_TEAM_ACTIONS:
            raise ValueError("a Team exposes too many Actions")
        return self

    def runtime_context(self) -> agent_runtime.TurnContext:
        return agent_runtime.TurnContext(
            thread_id=self.thread_id,
            team_name=self.team_name,
            assistants=tuple(
                agent_runtime.AssistantDefinition(
                    id=assistant.id,
                    genesis=assistant.genesis,
                    actions=tuple(
                        agent_runtime.ActionDefinition(
                            id=action.id,
                            summary=action.summary,
                            input_schema=action.input_schema,
                        )
                        for action in assistant.actions
                    ),
                )
                for assistant in self.assistants
            ),
            provider=agent_runtime.ProviderConfig(
                provider=self.provider.provider,
                model=self.provider.model,
                api_key=self.provider.api_key.get_secret_value(),
            ),
        )


class StartTurnInput(TurnContextInput):
    message: str = Field(min_length=1, max_length=agent_runtime.MAX_MESSAGE_CHARS)


class ResumeTurnInput(TurnContextInput):
    results: dict[str, Any] = Field(min_length=1, max_length=agent_runtime.MAX_TEAM_ACTIONS)


class DeleteThreadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1, max_length=256)

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, value: str) -> str:
        if agent_runtime.IDENTIFIER_RE.fullmatch(value) is None:
            raise ValueError("invalid conversation thread")
        return value


class ActionLabelsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderInput
    language_exemplar: str = Field(min_length=1, max_length=agent_runtime.MAX_LANGUAGE_EXEMPLAR_CHARS)
    actions: list[str] = Field(min_length=1, max_length=agent_runtime.MAX_ACTION_LABELS)

    @field_validator("language_exemplar", mode="before")
    @classmethod
    def normalize_language_exemplar(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return agent_runtime.normalize_language_exemplar(value)

    @field_validator("actions")
    @classmethod
    def validate_actions(cls, value: list[str]) -> list[str]:
        if (
            any(agent_runtime.ACTION_ID_RE.fullmatch(action_id) is None for action_id in value)
            or len(set(value)) != len(value)
        ):
            raise ValueError("invalid Action label ids")
        return value

    def runtime_provider(self) -> agent_runtime.ProviderConfig:
        return agent_runtime.ProviderConfig(
            provider=self.provider.provider,
            model=self.provider.model,
            api_key=self.provider.api_key.get_secret_value(),
        )


class RuntimeLike:
    """Structural documentation for the injected runtime used by the API and tests."""

    def start(self, context: agent_runtime.TurnContext, message: str) -> agent_runtime.TurnResult: ...

    def resume(
        self,
        context: agent_runtime.TurnContext,
        results: Mapping[str, object],
    ) -> agent_runtime.TurnResult: ...

    def delete_thread(self, thread_id: str) -> None: ...

    def action_labels(
        self,
        provider: agent_runtime.ProviderConfig,
        language_exemplar: str,
        action_ids: tuple[str, ...],
    ) -> tuple[agent_runtime.ActionLabel, ...]: ...


TokenReader = Callable[[], str]


class PruningSqliteSaver(SqliteSaver):
    """Retain only the latest self-contained checkpoint in each thread namespace."""

    def prune_thread(self, thread_id: str) -> None:
        with self.lock, self.conn:
            self.conn.execute(_PRUNE_WRITES_SQL, (thread_id, thread_id))
            self.conn.execute(_PRUNE_CHECKPOINTS_SQL, (thread_id, thread_id))


def _token_from_file() -> str:
    try:
        raw = TOKEN_FILE.read_bytes()
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Brain runtime authentication is unavailable") from exc
    if not 1 <= len(raw) <= MAX_TOKEN_BYTES:
        raise HTTPException(status_code=503, detail="Brain runtime authentication is unavailable")
    try:
        token = raw.decode().strip()
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=503, detail="Brain runtime authentication is unavailable") from exc
    if not token:
        raise HTTPException(status_code=503, detail="Brain runtime authentication is unavailable")
    return token


def _sqlite_runtime(path: Path = STATE_PATH) -> agent_runtime.AgentRuntime:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    connection = sqlite3.connect(path, check_same_thread=False)
    path.chmod(0o600)
    connection.execute("PRAGMA secure_delete=ON")
    checkpointer = PruningSqliteSaver(connection)
    checkpointer.setup()
    return agent_runtime.AgentRuntime(checkpointer)


def _response(result: agent_runtime.TurnResult) -> dict[str, object]:
    return {
        "status": result.status,
        "reply": result.reply,
        "actions": [
            {
                "interrupt_id": request.interrupt_id,
                "assistant_id": request.assistant_id,
                "action": request.action,
                "input": dict(request.input),
            }
            for request in result.actions
        ],
    }


def _action_labels_response(labels: tuple[agent_runtime.ActionLabel, ...]) -> dict[str, object]:
    return {"labels": [{"id": item.id, "label": item.label} for item in labels]}


def _close_owned_runtime(runtime: object, *, owned: bool) -> None:
    if not owned or runtime is None:
        return
    close = getattr(runtime, "close", None)
    if callable(close):
        close()


async def _state_error_response(_request, _exc: agent_runtime.RuntimeStateError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Brain runtime state operation failed"})


def create_app(
    *,
    runtime: RuntimeLike | None = None,
    token_reader: TokenReader = _token_from_file,
) -> FastAPI:
    owns_runtime = runtime is None

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        yield
        _close_owned_runtime(application.state.runtime, owned=owns_runtime)

    app = FastAPI(
        title="Shimpz Brain Runtime",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    app.state.runtime_lock = threading.Lock()
    app.add_exception_handler(agent_runtime.RuntimeStateError, _state_error_response)

    def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = token_reader()
        prefix = "Bearer "
        supplied = authorization[len(prefix) :] if authorization and authorization.startswith(prefix) else ""
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="Unauthorized")

    def current_runtime() -> RuntimeLike:
        if app.state.runtime is not None:
            return app.state.runtime
        with app.state.runtime_lock:
            if app.state.runtime is None:
                app.state.runtime = _sqlite_runtime()
        return app.state.runtime

    @app.exception_handler(agent_runtime.RuntimeContractError)
    async def contract_error(_request, exc: agent_runtime.RuntimeContractError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(agent_runtime.ProviderRequestError)
    async def provider_error(_request, _exc: agent_runtime.ProviderRequestError):
        return JSONResponse(status_code=502, content={"detail": "Model provider request failed"})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "runtime": "langgraph"}

    @app.post("/v1/turns", dependencies=[Depends(require_auth)])
    def start_turn(body: StartTurnInput) -> dict[str, object]:
        return _response(current_runtime().start(body.runtime_context(), body.message))

    @app.post("/v1/turns/resume", dependencies=[Depends(require_auth)])
    def resume_turn(body: ResumeTurnInput) -> dict[str, object]:
        return _response(current_runtime().resume(body.runtime_context(), body.results))

    @app.post("/v1/threads/delete", dependencies=[Depends(require_auth)])
    def delete_thread(body: DeleteThreadInput) -> dict[str, str]:
        current_runtime().delete_thread(body.thread_id)
        return {"status": "deleted"}

    @app.post("/v1/action-labels", dependencies=[Depends(require_auth)])
    def action_labels(body: ActionLabelsInput) -> dict[str, object]:
        labels = current_runtime().action_labels(
            body.runtime_provider(),
            body.language_exemplar,
            tuple(body.actions),
        )
        return _action_labels_response(labels)

    return app


app = create_app()
