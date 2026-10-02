"""Request-local attachment content on the real provider HTTP payloads (ADR-0093).

Both adapters run over an `httpx.MockTransport`, so the bodies asserted here are exactly what would be sent: no network,
key, or provider call is used. The checkpoint is inspected after every turn to prove attachment content never entered
graph state.
"""

from __future__ import annotations

import base64
import hashlib
import json
import unittest
from types import SimpleNamespace

import agent_runtime
import attachments as turn_attachments
import httpx
import runtime_api
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from test_runtime_api import TOKEN, body

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
PNG_BASE64 = base64.b64encode(PNG).decode()
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": False}
FILE_SCHEMA = {
    "type": "object",
    "properties": {"document": {"type": "string", "minLength": 32, "maxLength": 32, "pattern": "^[0-9a-f]{32}$"}},
    "required": ["document"],
    "additionalProperties": False,
}
ASSISTANT = agent_runtime.AssistantDefinition(
    "docs",
    "Docs stores and reads documents.",
    (
        agent_runtime.ActionDefinition("read-index", "Read the document index.", SCHEMA),
        agent_runtime.ActionDefinition(
            "store", "Store one document.", FILE_SCHEMA, authorization=True, input_files=("document",)
        ),
    ),
)


def _attachment(kind: str, file_id: str = "a" * 32) -> dict[str, object]:
    content: dict[str, object]
    if kind == "text":
        content = {"type": "text", "text": "Quarterly total: 1000", "pdf": True}
    elif kind == "image":
        content = {
            "type": "image",
            "media_type": "image/png",
            "width": 1,
            "height": 1,
            "base64": PNG_BASE64,
            "sha256": hashlib.sha256(PNG).hexdigest(),
        }
    else:
        content = {"type": "opaque", "reason": "unsupported"}
    return {
        "id": file_id,
        "name": f"{kind}.bin",
        "media_type": "application/octet-stream",
        "size": 10,
        "sha256": "f" * 64,
        "content": content,
    }


def _context(provider: str, model: str, *items: dict[str, object]) -> agent_runtime.TurnContext:
    return agent_runtime.TurnContext(
        "attachments-thread",
        "Docs Team",
        (ASSISTANT,),
        agent_runtime.ProviderConfig(provider, model, "sk-test-0123456789abcdef", "low"),
        memories=(),
        skills=(),
        routines=(),
        attachments=turn_attachments.admit(list(items)),
    )


class _Provider:
    """One provider behind a mock transport, recording each request body by path."""

    def __init__(self, provider: str, count: int = 300) -> None:
        self.provider = provider
        self.count = count
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if request.url.path.endswith("count_tokens") or request.url.path.endswith("input_tokens"):
            return httpx.Response(200, json={"input_tokens": self.count, "object": "response.input_tokens"})
        if self.provider == "anthropic":
            return httpx.Response(
                200,
                json={
                    "id": f"msg_{len(self.requests)}",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5-5",
                    "content": [{"type": "text", "text": "Read it."}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": f"resp_{len(self.requests)}",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-6.1-sol",
                "output": [
                    {
                        "type": "message",
                        "id": f"msg_{len(self.requests)}",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Read it.", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        )

    def runtime(self, saver: InMemorySaver) -> agent_runtime.AgentRuntime:
        client = httpx.Client(transport=httpx.MockTransport(self))
        return agent_runtime.AgentRuntime(
            saver, model_factory=lambda config: agent_runtime.provider_model(config, http_client=client)
        )

    def turns(self) -> list[dict]:
        return [body for path, body in self.requests if path in {"/v1/messages", "/v1/responses"}]

    def counts(self) -> list[dict]:
        return [body for path, body in self.requests if path.endswith(("count_tokens", "input_tokens"))]


def _checkpoint_text(saver: InMemorySaver) -> str:
    state = saver.get_tuple({"configurable": {"thread_id": "attachments-thread"}})
    return json.dumps(
        [message.model_dump(mode="json") for message in state.checkpoint["channel_values"]["messages"]], default=str
    ) + json.dumps(dict(state.metadata), default=str)


class ProviderPayloadTests(unittest.TestCase):
    def test_openai_receives_native_text_and_a_base64_image_only_in_the_request(self) -> None:
        provider = _Provider("openai")
        saver = InMemorySaver()
        runtime = provider.runtime(saver)
        result = runtime.start(
            _context("openai", "gpt-6.1-sol", _attachment("text"), _attachment("image", "b" * 32)),
            '{"files":[],"message":"Summarize the files"}',
        )
        self.assertEqual(result.reply, "Read it.")
        (turn,) = provider.turns()
        content = turn["input"][-1]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": '{"files":[],"message":"Summarize the files"}'})
        self.assertIn("PDF text only", content[1]["text"])
        self.assertIn(json.dumps("Quarterly total: 1000"), content[1]["text"])
        self.assertEqual(content[-1], {"type": "input_image", "image_url": f"data:image/png;base64,{PNG_BASE64}"})
        self.assertEqual(len(provider.counts()), 2)
        self.assertNotIn(PNG_BASE64, _checkpoint_text(saver))
        self.assertNotIn("Quarterly total", _checkpoint_text(saver))

    def test_anthropic_receives_a_native_base64_image_source(self) -> None:
        provider = _Provider("anthropic")
        saver = InMemorySaver()
        provider.runtime(saver).start(
            _context("anthropic", "claude-sonnet-5-5", _attachment("image")),
            '{"files":[],"message":"What is in the picture?"}',
        )
        (turn,) = provider.turns()
        image = turn["messages"][-1]["content"][-1]
        self.assertEqual(
            image, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_BASE64}}
        )
        self.assertEqual(provider.counts()[0]["messages"][0]["content"][-1]["type"], "image")
        self.assertNotIn(PNG_BASE64, _checkpoint_text(saver))

    def test_the_next_turn_forgets_the_attachment_exchange_whole(self) -> None:
        provider = _Provider("openai")
        saver = InMemorySaver()
        runtime = provider.runtime(saver)
        runtime.start(_context("openai", "gpt-6.1-sol", _attachment("text")), '{"files":[],"message":"Read it"}')
        runtime.start(_context("openai", "gpt-6.1-sol"), '{"files":[],"message":"And now?"}')
        second = provider.turns()[1]
        sent = json.dumps(second["input"])
        self.assertNotIn("Read it", sent)
        self.assertNotIn("Quarterly total", sent)
        self.assertIn("And now?", sent)


class _ActionProvider(_Provider):
    """An OpenAI provider whose first turn call requests the authorizing file Action."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        response = super().__call__(request)
        if request.url.path == "/v1/responses" and len(self.turns()) == 1:
            arguments = json.dumps({"document": "a" * 32})
            name = agent_runtime._tool_name("docs", "store")
            return httpx.Response(
                200,
                json={
                    **response.json(),
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": name,
                            "arguments": arguments,
                            "status": "completed",
                        }
                    ],
                },
            )
        return response


class ResumeTests(unittest.TestCase):
    def test_every_resume_carries_the_same_attachments_and_a_changed_set_is_refused(self) -> None:
        provider = _ActionProvider("openai")
        saver = InMemorySaver()
        runtime = provider.runtime(saver)
        context = _context("openai", "gpt-6.1-sol", _attachment("text"))
        paused = runtime.start(context, '{"files":[],"message":"Store the report"}')
        self.assertEqual(paused.status, "action-required")
        (request,) = paused.actions
        self.assertEqual((request.action, dict(request.input)), ("store", {"document": "a" * 32}))
        self.assertNotIn("Quarterly total", _checkpoint_text(saver))

        changed = _attachment("text")
        changed["content"]["text"] = "Quarterly total: 2000"
        with self.assertRaises(agent_runtime.RuntimeContractError):
            runtime.resume(_context("openai", "gpt-6.1-sol", changed), {request.interrupt_id: {"stored": True}})

        result = runtime.resume(context, {request.interrupt_id: {"stored": True}})
        self.assertEqual(result.reply, "Read it.")
        _first, second = provider.turns()
        self.assertIn("Quarterly total: 1000", json.dumps(second["input"]))
        self.assertEqual(len(provider.counts()), 1)
        self.assertNotIn("Quarterly total", _checkpoint_text(saver))


class ExposureTests(unittest.TestCase):
    def test_content_offers_only_authorizing_actions_and_no_learning_or_routines(self) -> None:
        provider = _Provider("openai")
        provider.runtime(InMemorySaver()).start(
            _context("openai", "gpt-6.1-sol", _attachment("text")), '{"files":[],"message":"Store it"}'
        )
        (turn,) = provider.turns()
        names = [tool["name"] for tool in turn["tools"]]
        self.assertEqual(len([name for name in names if name.startswith("a_docs")]), 1)
        self.assertTrue(any("store" in name for name in names))
        self.assertFalse(any("read-index" in name or "read_index" in name for name in names))
        self.assertNotIn("team_memory", " ".join(names))
        instructions = turn["instructions"] if "instructions" in turn else json.dumps(turn["input"][0])
        self.assertIn("file_input", instructions)
        self.assertNotIn("read-index", instructions)

    def test_an_opaque_file_keeps_every_action(self) -> None:
        provider = _Provider("openai")
        provider.runtime(InMemorySaver()).start(
            _context("openai", "gpt-6.1-sol", _attachment("opaque")), '{"files":[],"message":"Store it"}'
        )
        names = [tool["name"] for tool in provider.turns()[0]["tools"]]
        self.assertEqual(len([name for name in names if name.startswith("a_docs")]), 2)

    def test_knowledge_tools_are_withheld_while_files_are_attached(self) -> None:
        context = _context("openai", "gpt-6.1-sol", _attachment("opaque"))
        self.assertEqual(agent_runtime._knowledge_tools(context), (False, False))
        self.assertEqual(agent_runtime._knowledge_tools(_context("openai", "gpt-6.1-sol")), (True, True))


class ChargeTests(unittest.TestCase):
    def test_a_file_over_its_counted_ceiling_refuses_the_turn_before_any_model_call(self) -> None:
        provider = _Provider("openai", count=turn_attachments.MAX_FILE_TOKENS + 1)
        with self.assertRaises(agent_runtime.RuntimeContractError):
            provider.runtime(InMemorySaver()).start(
                _context("openai", "gpt-6.1-sol", _attachment("text")), '{"files":[],"message":"Read"}'
            )
        self.assertEqual(provider.turns(), [])

    def test_a_count_failure_falls_back_to_a_conservative_estimate(self) -> None:
        attachment = turn_attachments.admit([_attachment("text")])[0]

        def failing(_blocks: list, _timeout: float) -> int:
            raise turn_attachments.CountUnavailableError("down")

        estimated = turn_attachments.charges((attachment,), failing)
        self.assertEqual(estimated, (turn_attachments.estimated_charge(attachment),))
        self.assertEqual(turn_attachments.charges((attachment,), lambda _blocks, _timeout: 42), (42,))
        image = turn_attachments.admit([_attachment("image")])[0]
        self.assertEqual(turn_attachments.charges((image,), None), (turn_attachments.IMAGE_TOKEN_CEILING,))

    def test_call_and_turn_ceilings(self) -> None:
        with self.assertRaises(turn_attachments.AttachmentContractError):
            turn_attachments.admit_charges((9_000,))
        with self.assertRaises(turn_attachments.AttachmentContractError):
            turn_attachments.admit_charges((8_000, 8_001))
        self.assertEqual(turn_attachments.admit_charges((8_000, 8_000)), 16_000)
        turn_attachments.admit_call(16_000, 3)
        with self.assertRaises(turn_attachments.AttachmentContractError):
            turn_attachments.admit_call(16_000, 4)
        turn_attachments.admit_call(0, 1_000)


class AdmissionTests(unittest.TestCase):
    def test_only_the_closed_bounded_shapes_are_admitted(self) -> None:
        self.assertEqual(len(turn_attachments.admit([_attachment("text"), _attachment("image", "b" * 32)])), 2)
        wrong_digest = _attachment("image")
        wrong_digest["content"]["sha256"] = "0" * 64
        too_wide = _attachment("image")
        too_wide["content"]["width"] = 1_569
        gif = _attachment("image")
        gif["content"]["media_type"] = "image/gif"
        extra = {**_attachment("text"), "path": "secret"}
        empty = _attachment("text")
        empty["content"]["text"] = "   "
        reason = _attachment("opaque")
        reason["content"]["reason"] = "maybe"
        five_images = [_attachment("image", f"{index:032x}") for index in range(5)]
        for raw in (
            [wrong_digest],
            [too_wide],
            [gif],
            [extra],
            [empty],
            [reason],
            [_attachment("text"), _attachment("text")],
            five_images,
            [_attachment("text", "A" * 32)],
            "not a list",
        ):
            with self.subTest(raw=str(raw)[:60]), self.assertRaises(turn_attachments.AttachmentContractError):
                turn_attachments.admit(raw)

    def test_the_commitment_binds_the_exact_prepared_content(self) -> None:
        first = turn_attachments.admit([_attachment("text")])
        changed = _attachment("text")
        changed["content"]["text"] = "Quarterly total: 2000"
        self.assertNotEqual(
            turn_attachments.commitment(first), turn_attachments.commitment(turn_attachments.admit([changed]))
        )
        self.assertEqual(
            turn_attachments.commitment(first),
            turn_attachments.commitment(turn_attachments.admit([_attachment("text")])),
        )


class TurnEndpointTests(unittest.TestCase):
    def test_start_and_resume_admit_attachments_and_refuse_an_invalid_field(self) -> None:
        contexts: list[agent_runtime.TurnContext] = []
        runtime = SimpleNamespace(
            start=lambda context, _message, _conversation=(): (
                contexts.append(context) or agent_runtime.TurnResult(status="completed", reply="Done.")
            ),
            resume=lambda context, _results: (
                contexts.append(context) or agent_runtime.TurnResult(status="completed", reply="Done.")
            ),
        )
        api = TestClient(runtime_api.create_app(runtime=runtime, token_reader=lambda: TOKEN))
        headers = {"Authorization": f"Bearer {TOKEN}"}
        attached = body(attachments=[_attachment("image")])
        self.assertEqual(api.post("/v1/turns", json=attached, headers=headers).status_code, 200)
        resume = {key: value for key, value in attached.items() if key not in {"message", "conversation", "locale"}}
        resume["results"] = {"interrupt-1": {"ok": True}}
        self.assertEqual(api.post("/v1/turns/resume", json=resume, headers=headers).status_code, 200)
        self.assertEqual([len(context.attachments) for context in contexts], [1, 1])
        bad = _attachment("image")
        bad["content"]["sha256"] = "0" * 64
        refused = api.post("/v1/turns", json=body(attachments=[bad]), headers=headers)
        self.assertEqual((refused.status_code, refused.json()), (400, {"detail": "invalid attachments"}))
        missing = body()
        del missing["attachments"]
        self.assertEqual(api.post("/v1/turns", json=missing, headers=headers).status_code, 422)
        self.assertEqual(len(contexts), 2)


if __name__ == "__main__":
    unittest.main()
