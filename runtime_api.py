"""Authenticated HTTP boundary for the isolated Shimpz LangGraph runtime."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import hmac
import os
import sqlite3
import threading
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import action_labels
import action_purpose
import agent_runtime
import attachments as turn_attachments
import capability_plan
import intent_route
import interface_language
import memory as team_memory
import model_usage
import provider_cancel
import routine_recovery
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, ConfigDict, Field, SecretStr, StrictBool, field_validator, model_validator

import routine as team_routine

TOKEN_FILE = Path(os.environ.get("SHIMPZ_BRAIN_RUNTIME_TOKEN_FILE", "/run/shimpz-brain-runtime/token"))
STATE_PATH = Path(os.environ.get("SHIMPZ_BRAIN_RUNTIME_STATE", "/var/lib/shimpz-brain-runtime/checkpoints.sqlite3"))
MAX_TOKEN_BYTES = 4 * 1024
CAPABILITY_PLAN_CONCURRENCY = 2
INTENT_ROUTE_CONCURRENCY = 2
# Team sends compact canonical JSON, and a start or resume succeeds only when what the model window counts (genesis,
# Assistant and Action ids and summaries, input schemas, start-time knowledge, and the message or Action results) fits
# the smallest window: (1,000,000 - 128,000) tokens of 3 bytes, about 2.5 MiB. What a canonical request carries beyond
# that (resumed knowledge, skill contracts, the conversation window, provider settings, identifiers, and framing) stays
# under 0.6 MiB, and the other endpoints carry far less. A larger body could only fail after it was decoded, so it is
# refused before it is read.
MAX_REQUEST_BYTES = 4 * 1024 * 1024
# Team writes one whole body at once over the private network; a body still incomplete after this is refused.
REQUEST_BODY_SECONDS = 30.0
# Each admitted body reserves memory until its request has completed and its garbage is collected; a body that would
# exceed the budget is refused before any byte of it is read. Python holds every decoded JSON value as an object, so
# dense data costs far more than its bytes: through this app and a whole turn, depth-32 nested lists in Action input
# schemas grew peak RSS by 118 bytes per body byte (tool conversion copies them), dense Action results by 65, and JSON
# of any shape by 53 in decoding alone. A body therefore reserves 128 bytes per byte of its declared length, or of the
# whole ceiling when it streams, plus 48 MiB for what its length does not measure: the history a turn restores from
# its checkpoint (about 30 MiB for a near-window pending turn), provider serialization, and validation errors, which
# ClosedInput bounds by the objects a body's lists admit. The budget admits one largest request beside four small
# ones, or fifteen small ones, and leaves the rest of the 1 GiB container to the loaded process (about 140 MiB). In a
# warmed process with the image's allocator settings, four dense starts at once peaked at 530 MiB and fifteen resumes of
# near-window pending turns at 540 MiB. These are measured bounds for this code and its dependencies, not proofs.
MEMORY_PER_REQUEST = 48 * 1024 * 1024
MEMORY_PER_BODY_BYTE = 128
MAX_REQUEST_MEMORY = 768 * 1024 * 1024
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
    api_key: SecretStr = Field(min_length=1, max_length=16 * 1024)


class ChatProviderInput(ProviderInput):
    effort: Literal["low", "medium", "high"]


class ActionInput(ClosedInput):
    id: str = Field(min_length=1, max_length=128)
    summary: str = Field(min_length=1, max_length=2_000)
    input_schema: dict[str, Any]
    # Whether the Action declares an authorization capability, and which input properties take a file (ADR-0093).
    authorization: StrictBool
    input_files: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(max_length=1)


class AssistantInput(ClosedInput):
    id: str = Field(min_length=1, max_length=128)
    genesis: str = Field(min_length=1, max_length=agent_runtime.MAX_GENESIS_BYTES)
    actions: list[ActionInput] = Field(max_length=agent_runtime.MAX_ACTIONS_PER_ASSISTANT)


class TurnContextInput(ClosedInput):
    thread_id: str = Field(min_length=1, max_length=256)
    team_name: str = Field(min_length=1, max_length=agent_runtime.MAX_TEAM_NAME_CHARS)
    assistants: list[AssistantInput] = Field(max_length=agent_runtime.MAX_ASSISTANTS)
    provider: ChatProviderInput
    # What the Team remembers (ADR-0084); null where memory is unavailable, which also withholds the memory tool.
    memories: Annotated[list[dict[str, Any]], Field(max_length=team_memory.MAX_MEMORIES)] | None
    # The procedures the Team learned (ADR-0085); null where learning is unavailable.
    skills: Annotated[list[dict[str, Any]], Field(max_length=team_memory.MAX_SKILLS)] | None
    # The Team's Routines as data (ADR-0086); null withholds the Routine tool.
    routines: Annotated[list[dict[str, Any]], Field(max_length=team_routine.MAX_ROUTINES)] | None
    # False in a Routine run, whose knowledge is read-only.
    knowledge_writable: StrictBool
    # The message's prepared files (ADR-0093), resent with every resume; request-local model content only.
    attachments: list[dict[str, Any]] = Field(max_length=turn_attachments.MAX_ATTACHMENTS)

    @field_validator("team_name", mode="before")
    @classmethod
    def normalize_team_name(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return agent_runtime.normalize_team_name(value)

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
                            authorization=action.authorization,
                            input_files=tuple(action.input_files),
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
                effort=self.provider.effort,
            ),
            memories=_memories(self.memories),
            skills=None if self.skills is None else tuple(self.skills),
            routines=None if self.routines is None else tuple(self.routines),
            knowledge_writable=self.knowledge_writable,
            attachments=_attachments(self.attachments),
        )


def _attachments(value: list[dict[str, Any]]) -> tuple[turn_attachments.Attachment, ...]:
    try:
        return turn_attachments.admit(value)
    except turn_attachments.AttachmentContractError as exc:
        raise agent_runtime.RuntimeContractError("invalid attachments") from exc


def _memories(value: list[dict[str, Any]] | None) -> tuple[team_memory.Memory, ...] | None:
    if value is None:
        return None
    try:
        return team_memory.canonical(value)
    except team_memory.MemoryContractError as exc:
        raise agent_runtime.RuntimeContractError("invalid memory") from exc


class ConversationEntryInput(ClosedInput):
    model_config = ConfigDict(strict=True)

    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=intent_route.MAX_CONVERSATION_TEXT_CHARS)
    truncated: bool

    def runtime_entry(self) -> intent_route.ConversationEntry:
        return intent_route.ConversationEntry(self.role, self.text, self.truncated)


class StartTurnInput(TurnContextInput):
    message: str = Field(min_length=1, max_length=agent_runtime.MAX_MESSAGE_CHARS)
    # The interface language this logical turn writes in (ADR-0090); null follows the user's message. Its start pins it.
    locale: interface_language.Locale | None
    # Eight entries of at most 512 characters cannot exceed the 4,096-character window total.
    conversation: list[ConversationEntryInput] = Field(max_length=intent_route.MAX_CONVERSATION_ENTRIES)

    def runtime_conversation(self) -> tuple[intent_route.ConversationEntry, ...]:
        return tuple(entry.runtime_entry() for entry in self.conversation)

    def runtime_context(self) -> agent_runtime.TurnContext:
        return replace(super().runtime_context(), locale=self.locale)


class ResumeTurnInput(TurnContextInput):
    results: dict[str, Any] = Field(min_length=1, max_length=agent_runtime.MAX_ACTION_RESULTS)


class ActionPurposeInput(ClosedInput):
    """One exact pending Action interrupt and the reviewed names its human request shows (ADR-0090)."""

    thread_id: str = Field(min_length=1, max_length=256)
    interrupt_id: str = Field(min_length=1, max_length=256)
    assistant_id: str = Field(min_length=1, max_length=128)
    assistant_name: str = Field(min_length=1, max_length=action_purpose.MAX_ASSISTANT_NAME_CHARS)
    action_id: str = Field(min_length=1, max_length=128)
    action_summary: str = Field(min_length=1, max_length=action_purpose.MAX_ACTION_SUMMARY_CHARS)
    provider: ProviderInput

    def runtime_provider(self) -> agent_runtime.ProviderConfig:
        return agent_runtime.ProviderConfig(
            provider=self.provider.provider,
            model=self.provider.model,
            api_key=self.provider.api_key.get_secret_value(),
        )

    def pending_action(self) -> action_purpose.PendingAction:
        return action_purpose.PendingAction(
            thread_id=self.thread_id,
            interrupt_id=self.interrupt_id,
            assistant_id=self.assistant_id,
            action_id=self.action_id,
            assistant_name=self.assistant_name,
            action_summary=self.action_summary,
        )


class DeleteThreadInput(ClosedInput):
    thread_id: str = Field(min_length=1, max_length=256)

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, value: str) -> str:
        if agent_runtime.IDENTIFIER_RE.fullmatch(value) is None:
            raise ValueError("invalid conversation thread")
        return value


class ActionLabelsInput(ClosedInput):
    provider: ProviderInput
    locale: interface_language.Locale
    actions: list[str] = Field(min_length=1, max_length=action_labels.MAX_ACTION_LABELS)

    @field_validator("actions")
    @classmethod
    def validate_actions(cls, value: list[str]) -> list[str]:
        if any(agent_runtime.ACTION_ID_RE.fullmatch(action_id) is None for action_id in value) or len(
            set(value)
        ) != len(value):
            raise ValueError("invalid Action label ids")
        return value

    def runtime_provider(self) -> agent_runtime.ProviderConfig:
        return agent_runtime.ProviderConfig(
            provider=self.provider.provider,
            model=self.provider.model,
            api_key=self.provider.api_key.get_secret_value(),
        )


class RoutineInput(ClosedInput):
    name: str = Field(min_length=1, max_length=80)
    request: str = Field(min_length=1, max_length=500)


class RoutineStepInput(ClosedInput):
    assistant: str = Field(min_length=1, max_length=128)
    action: str = Field(min_length=1, max_length=128)


class RoutineRecoveryInput(ClosedInput):
    """A held run's failed step, Team's proof that it had no effect, and its sanitized diagnostics as data."""

    provider: ProviderInput
    locale: interface_language.Locale | None
    routine: RoutineInput
    step: RoutineStepInput
    proof: Literal["not_occurred", "no_effect"]
    diagnostics: list[dict[str, Any]] = Field(max_length=routine_recovery.MAX_DIAGNOSTICS)

    def runtime_provider(self) -> agent_runtime.ProviderConfig:
        return agent_runtime.ProviderConfig(
            provider=self.provider.provider,
            model=self.provider.model,
            api_key=self.provider.api_key.get_secret_value(),
        )

    def runtime_request(self) -> routine_recovery.RecoveryRequest:
        return routine_recovery.RecoveryRequest(
            self.routine.name,
            self.routine.request,
            self.step.assistant,
            self.step.action,
            self.proof,
            tuple(self.diagnostics),
            self.locale,
        )


class CapabilityIntegrationInput(ClosedInput):
    id: str = Field(min_length=1, max_length=128)
    provider: str = Field(min_length=1, max_length=128)


class CapabilityCandidateInput(ClosedInput):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=capability_plan.MAX_NAME_CHARS)
    summary: str = Field(min_length=1, max_length=capability_plan.MAX_SUMMARY_CHARS)
    actions: list[str] = Field(min_length=1, max_length=capability_plan.MAX_ACTIONS)
    integrations: list[CapabilityIntegrationInput] = Field(max_length=capability_plan.MAX_INTEGRATIONS)

    def runtime_candidate(self) -> capability_plan.CapabilityCandidate:
        return capability_plan.CapabilityCandidate(
            id=self.id,
            name=self.name,
            summary=self.summary,
            actions=tuple(self.actions),
            integrations=tuple(
                capability_plan.CapabilityIntegration(id=item.id, provider=item.provider) for item in self.integrations
            ),
        )


class CapabilityPlanInput(ClosedInput):
    provider: ProviderInput
    objective: str = Field(min_length=1, max_length=capability_plan.MAX_OBJECTIVE_CHARS)
    candidates: list[CapabilityCandidateInput] = Field(
        min_length=1,
        max_length=capability_plan.MAX_CANDIDATES,
    )

    def runtime_provider(self) -> agent_runtime.ProviderConfig:
        return agent_runtime.ProviderConfig(
            provider=self.provider.provider,
            model=self.provider.model,
            api_key=self.provider.api_key.get_secret_value(),
        )

    def runtime_candidates(self) -> tuple[capability_plan.CapabilityCandidate, ...]:
        return tuple(item.runtime_candidate() for item in self.candidates)


class DirectoryCandidateInput(ClosedInput):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=intent_route.MAX_NAME_CHARS)
    summary: str = Field(max_length=intent_route.MAX_SUMMARY_CHARS)

    def runtime_candidate(self) -> intent_route.DirectoryCandidate:
        return intent_route.DirectoryCandidate(id=self.id, name=self.name, summary=self.summary)


class LifecycleReferenceInput(ClosedInput):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=intent_route.MAX_NAME_CHARS)

    def runtime_reference(self) -> intent_route.LifecycleReference:
        return intent_route.LifecycleReference(id=self.id, name=self.name)


class DecisionProviderInput(ClosedInput):
    """The Supervisor's request-scoped TypeSafe key for the classification fast path."""

    provider: Literal["typesafe"]
    api_key: SecretStr = Field(min_length=16, max_length=8192)


class IntentRouteInput(ClosedInput):
    provider: ProviderInput
    decision_provider: DecisionProviderInput | None = None
    objective: str = Field(min_length=1, max_length=intent_route.MAX_OBJECTIVE_CHARS)
    expected_intent: Literal["assistant-install", "assistant-uninstall"] | None
    candidates: list[DirectoryCandidateInput] = Field(max_length=intent_route.MAX_CANDIDATES)
    lifecycle_reference: LifecycleReferenceInput | None
    conversation: list[ConversationEntryInput] = Field(max_length=intent_route.MAX_CONVERSATION_ENTRIES)
    # The interface language the presentation-only reply is written in (ADR-0090).
    locale: interface_language.Locale

    @model_validator(mode="after")
    def validate_lifecycle_context(self) -> Self:
        if self.expected_intent is not None and (
            self.lifecycle_reference is not None or self.conversation or self.decision_provider is not None
        ):
            raise ValueError("selection cannot include lifecycle state or a decision provider")
        if sum(len(entry.text) for entry in self.conversation) > intent_route.MAX_CONVERSATION_CHARS:
            raise ValueError("conversation window is too large")
        return self

    def runtime_provider(self) -> agent_runtime.ProviderConfig:
        return agent_runtime.ProviderConfig(
            provider=self.provider.provider,
            model=self.provider.model,
            api_key=self.provider.api_key.get_secret_value(),
        )

    def runtime_candidates(self) -> tuple[intent_route.DirectoryCandidate, ...]:
        return tuple(item.runtime_candidate() for item in self.candidates)

    def runtime_context(self) -> intent_route.LifecycleContext | None:
        reference = None if self.lifecycle_reference is None else self.lifecycle_reference.runtime_reference()
        conversation = tuple(entry.runtime_entry() for entry in self.conversation)
        if reference is None and not conversation:
            return None
        return intent_route.LifecycleContext(reference, conversation)


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
        locale: str,
        action_ids: tuple[str, ...],
    ) -> tuple[action_labels.ActionLabel, ...]: ...

    def action_purpose(
        self,
        provider: agent_runtime.ProviderConfig,
        pending: action_purpose.PendingAction,
    ) -> str | None: ...

    def capability_plan(
        self,
        provider: agent_runtime.ProviderConfig,
        objective: str,
        candidates: tuple[capability_plan.CapabilityCandidate, ...],
    ) -> capability_plan.CapabilityPlan: ...

    def routine_recovery(
        self, provider: agent_runtime.ProviderConfig, request: routine_recovery.RecoveryRequest
    ) -> str: ...

    def intent_route(
        self,
        provider: agent_runtime.ProviderConfig,
        objective: str,
        expected_intent: intent_route.LifecycleIntent | None,
        candidates: tuple[intent_route.DirectoryCandidate, ...],
        context: intent_route.LifecycleContext | None,
        locale: str,
        decision_key: str | None = None,
    ) -> intent_route.IntentRoute: ...


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


class _BodyTooLargeError(Exception):
    """The request body grew past the ceiling while it was being received."""


async def _refuse_body(scope, receive, send, status_code: int, detail: str, **headers: str) -> None:
    response = JSONResponse(
        status_code=status_code, content={"detail": detail}, headers={"Connection": "close", **headers}
    )
    await response(scope, receive, send)


def _authenticate(authorization: str | None, token_reader: TokenReader) -> None:
    """The one bearer check: the private runtime token, compared in constant time, or 401."""
    expected = token_reader()
    prefix = "Bearer "
    supplied = authorization[len(prefix) :] if authorization and authorization.startswith(prefix) else ""
    if not supplied or not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _reservation(headers: dict[bytes, bytes], max_bytes: int) -> int:
    """The bytes a request may buffer: zero without a body, its declared length, or the ceiling when it streams."""
    declared = headers.get(b"content-length")
    if declared is None:
        return max_bytes if b"transfer-encoding" in headers else 0
    return int(declared) if declared.isdigit() else max_bytes


class BoundedBody:
    """Admit and receive each whole request body within byte, memory, and time bounds before anything parses it.

    A body reserves a fixed amount of memory plus a multiple of its length for its whole request, and the reservation
    returns to the budget only after a full collection has freed what the request left behind: a turn leaves cyclic
    garbage that would otherwise stay resident until some later collection, unaccounted for.

    FastAPI buffers and decodes a body before the bearer dependency runs, so a request with a body must first pass
    the same bearer check here: no byte of an unauthenticated body is read or decoded. A bodyless request, such as
    health, needs neither admission nor this early check; its route's own dependency still applies.
    """

    def __init__(
        self,
        app,
        *,
        authenticate: Callable[[str | None], None],
        max_bytes: int,
        memory_per_request: int,
        memory_per_body_byte: int,
        memory_budget: int,
        deadline_seconds: float,
    ) -> None:
        self.app = app
        self.authenticate = authenticate
        self.max_bytes = max_bytes
        self.memory_per_request = memory_per_request
        self.memory_per_body_byte = memory_per_body_byte
        self.memory_budget = memory_budget
        self.deadline_seconds = deadline_seconds
        # Only the event loop thread changes this, with no await between its check and its update.
        self.reserved = 0

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        size = _reservation(headers, self.max_bytes)
        if size == 0:
            await self.app(scope, receive, send)
            return
        authorization = headers.get(b"authorization")
        try:
            self.authenticate(None if authorization is None else authorization.decode("latin-1"))
        except HTTPException as exc:
            await _refuse_body(scope, receive, send, exc.status_code, exc.detail)
            return
        memory = self.memory_per_request + size * self.memory_per_body_byte
        if size > self.max_bytes:
            await _refuse_body(scope, receive, send, 413, "Request body is too large")
        elif self.reserved + memory > self.memory_budget:
            await _refuse_body(
                scope, receive, send, 503, "Brain runtime request capacity reached", **{"Retry-After": "1"}
            )
        else:
            self.reserved += memory
            try:
                await self._admitted(scope, receive, send, size)
            finally:
                gc.collect()
                self.reserved -= memory

    async def _admitted(self, scope, receive, send, size: int) -> None:
        """Receive the admitted body within its reservation and the deadline, then hand it on exactly once."""
        try:
            async with asyncio.timeout(self.deadline_seconds):
                body = await self._receive_body(receive, size)
        except TimeoutError:
            await _refuse_body(scope, receive, send, 408, "Request body was not received in time")
            return
        except _BodyTooLargeError:
            await _refuse_body(scope, receive, send, 413, "Request body is too large")
            return
        if body is None:
            return
        delivered = False

        async def replay() -> dict[str, Any]:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)

    @staticmethod
    async def _receive_body(receive, limit: int) -> bytes | None:
        """Return the whole body, or None when the client disconnected before sending it."""
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return None
            body += message.get("body", b"")
            if len(body) > limit:
                raise _BodyTooLargeError
            if not message.get("more_body", False):
                return bytes(body)


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
        "clarification": None if result.clarification is None else result.clarification.to_dict(),
        "memory": [change.to_dict() for change in result.memory],
        "routine": result.routine,
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


def _action_labels_response(labels: tuple[action_labels.ActionLabel, ...]) -> dict[str, object]:
    return {"labels": [{"id": item.id, "label": item.label} for item in labels]}


def _capability_plan_response(plan: capability_plan.CapabilityPlan) -> dict[str, object]:
    return {"status": plan.status, "assistant_ids": list(plan.assistant_ids)}


def _intent_route_response(route: intent_route.IntentRoute) -> dict[str, object]:
    return {
        "intent": route.intent,
        "query": route.query,
        "assistant_ids": list(route.assistant_ids),
        "reply": route.reply,
        "task_follows": route.task_follows,
    }


def _run_capability_plan(
    runtime: RuntimeLike,
    slots: threading.BoundedSemaphore,
    body: CapabilityPlanInput,
) -> dict[str, object]:
    if not slots.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Capability planner capacity reached")
    try:
        plan, usage = model_usage.measure(
            lambda: runtime.capability_plan(
                body.runtime_provider(),
                body.objective,
                body.runtime_candidates(),
            )
        )
        return {**_capability_plan_response(plan), "usage": usage}
    finally:
        slots.release()


def _run_intent_route(
    runtime: RuntimeLike,
    slots: threading.BoundedSemaphore,
    body: IntentRouteInput,
) -> dict[str, object]:
    if not slots.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Intent route capacity reached")
    try:
        decision = body.decision_provider
        route, usage = model_usage.measure(
            lambda: runtime.intent_route(
                body.runtime_provider(),
                body.objective,
                body.expected_intent,
                body.runtime_candidates(),
                body.runtime_context(),
                body.locale,
                decision_key=None if decision is None else decision.api_key.get_secret_value(),
            )
        )
        return {**_intent_route_response(route), "usage": usage}
    finally:
        slots.release()


def _close_owned_runtime(runtime: object, *, owned: bool) -> None:
    if not owned or runtime is None:
        return
    close = getattr(runtime, "close", None)
    if callable(close):
        close()


async def _validation_error_response(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """Say only where and why a request is malformed; its own values, such as a provider key, are never echoed."""
    errors = [{"loc": list(error["loc"]), "type": error["type"], "msg": error["msg"]} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


async def _state_error_response(_request, _exc: agent_runtime.RuntimeStateError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Brain runtime state operation failed"})


def _register_routine_recovery(app: FastAPI, current_runtime: Callable[[], RuntimeLike], require_auth) -> None:
    """The one model decision of a held Routine run's automatic recovery, with no tools or history (ADR-0092)."""

    @app.post("/v1/routine-recovery", dependencies=[Depends(require_auth)])
    async def routine_recovery_decision(request: Request, body: RoutineRecoveryInput) -> dict[str, object]:
        # Team's Stop or recovery deadline closes its request; that cancels only this decision's provider I/O.
        decision, usage = await _cancellable(
            request,
            lambda: current_runtime().routine_recovery(body.runtime_provider(), body.runtime_request()),
            "Routine recovery cancelled",
        )
        return {"decision": decision, "usage": usage}


def _register_intent_route(
    app: FastAPI,
    current_runtime: Callable[[], RuntimeLike],
    require_auth: Callable[[], None],
) -> None:
    """Register the independent structured-decision capacity boundary."""

    @app.post("/v1/intent-route", dependencies=[Depends(require_auth)])
    def create_intent_route(body: IntentRouteInput) -> dict[str, object]:
        return _run_intent_route(
            current_runtime(),
            app.state.intent_route_slots,
            body,
        )


async def _cancel_on_disconnect(request: Request, scope: provider_cancel.CancelScope) -> None:
    while (await request.receive())["type"] != "http.disconnect":
        pass
    scope.cancel()


async def _cancellable[T](request: Request, work: Callable[[], T], cancelled: str) -> tuple[T, dict[str, object]]:
    """Run one synchronous operation to completion; a Team disconnect cancels only its provider I/O (ADR-0079)."""
    scope = provider_cancel.CancelScope()
    watcher = asyncio.create_task(_cancel_on_disconnect(request, scope))
    try:
        return await run_in_threadpool(scope.run, lambda: model_usage.measure(work))
    except provider_cancel.ProviderCallCancelled:
        raise HTTPException(status_code=409, detail=cancelled) from None
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher


async def _cancellable_turn(request: Request, work: Callable[[], agent_runtime.TurnResult]) -> dict[str, object]:
    result, usage = await _cancellable(request, work, "Chat turn cancelled")
    return {**_response(result), "usage": usage}


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
    app.state.capability_plan_slots = threading.BoundedSemaphore(CAPABILITY_PLAN_CONCURRENCY)
    app.state.intent_route_slots = threading.BoundedSemaphore(INTENT_ROUTE_CONCURRENCY)
    app.add_exception_handler(agent_runtime.RuntimeStateError, _state_error_response)
    app.add_exception_handler(RequestValidationError, _validation_error_response)
    app.add_middleware(
        BoundedBody,
        authenticate=lambda authorization: _authenticate(authorization, token_reader),
        max_bytes=MAX_REQUEST_BYTES,
        memory_per_request=MEMORY_PER_REQUEST,
        memory_per_body_byte=MEMORY_PER_BODY_BYTE,
        memory_budget=MAX_REQUEST_MEMORY,
        deadline_seconds=REQUEST_BODY_SECONDS,
    )

    def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
        _authenticate(authorization, token_reader)

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

    # Constant and non-blocking, so it stays off the worker threads that blocking turns can saturate.
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "runtime": "langgraph"}

    @app.post("/v1/turns", dependencies=[Depends(require_auth)])
    async def start_turn(request: Request, body: StartTurnInput) -> dict[str, object]:
        return await _cancellable_turn(
            request,
            lambda: current_runtime().start(body.runtime_context(), body.message, body.runtime_conversation()),
        )

    # Why a pending Action pauses for a person (ADR-0090); a Team disconnect cancels its provider call.
    @app.post("/v1/turns/purpose", dependencies=[Depends(require_auth)])
    async def action_purpose(request: Request, body: ActionPurposeInput) -> dict[str, object]:
        purpose, usage = await _cancellable(
            request,
            lambda: current_runtime().action_purpose(body.runtime_provider(), body.pending_action()),
            "Action purpose cancelled",
        )
        return {"purpose": purpose, "usage": usage}

    @app.post("/v1/turns/resume", dependencies=[Depends(require_auth)])
    async def resume_turn(request: Request, body: ResumeTurnInput) -> dict[str, object]:
        return await _cancellable_turn(request, lambda: current_runtime().resume(body.runtime_context(), body.results))

    @app.post("/v1/threads/delete", dependencies=[Depends(require_auth)])
    def delete_thread(body: DeleteThreadInput) -> dict[str, str]:
        current_runtime().delete_thread(body.thread_id)
        return {"status": "deleted"}

    @app.post("/v1/action-labels", dependencies=[Depends(require_auth)])
    def action_labels(body: ActionLabelsInput) -> dict[str, object]:
        labels, usage = model_usage.measure(
            lambda: current_runtime().action_labels(
                body.runtime_provider(),
                body.locale,
                tuple(body.actions),
            )
        )
        return {**_action_labels_response(labels), "usage": usage}

    @app.post("/v1/capability-plan", dependencies=[Depends(require_auth)])
    def create_capability_plan(body: CapabilityPlanInput) -> dict[str, object]:
        return _run_capability_plan(
            current_runtime(),
            app.state.capability_plan_slots,
            body,
        )

    _register_intent_route(app, current_runtime, require_auth)
    _register_routine_recovery(app, current_runtime, require_auth)

    return app


app = create_app()
