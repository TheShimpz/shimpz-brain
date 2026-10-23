"""The Browser agent's stateless calls: a closed request, untrusted page data, and canonical answers."""

import base64
import itertools
import json
import unittest
from unittest import mock

import agent_runtime
import browser_step
import browser_thread
import provider_client
from browser_step import (
    BROWSER_MODELS,
    MAX_PRICED_INPUT_TOKENS,
    PNG_SIGNATURE,
    StepAnswer,
    ThreadAnswer,
    _prompt,
    check,
    final,
    fold,
    input_bound,
    model,
    step,
    validate,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from protocol.team.http.v1 import browser as browser_protocol
from runtime_errors import ProviderResponseError, RuntimeContractError
from structured_fake import StructuredFakeModel

EMPTY = {"folded": [], "folded_omitted": 0, "notes": [], "notes_omitted": 0, "steps": []}
ENTRY = {
    "step": {"kind": "navigate", "url": "https://example.com"},
    "outcome": "ok",
    "page": {"url": "about:blank", "title": ""},
    "text": "url: about:blank",
    "downloads": None,
    "screenshot": False,
}
REQUEST = {
    "objective": "Veja o prazo do edital.",
    "language": "pt-BR",
    "thread": {**EMPTY, "steps": [ENTRY]},
    "observation": 'url: https://example.com\ntitle: Example\nactions:\np1:0 link "Edital" (click)\npage:\n- heading',
    "screenshot": None,
    "downloads": {"files": ["report.csv"], "pending": 1, "truncated": False},
}
NULLS = {
    "url": None,
    "ref": None,
    "refs": None,
    "text": None,
    "direction": None,
    "summary": None,
    "reason": None,
    "sentence": None,
    "note": None,
}


def _answer(**fields):
    return AIMessage(content=json.dumps({**NULLS, **fields}))


def _haiku(**fields):
    return AIMessage(content=json.dumps({**{k: v for k, v in NULLS.items() if k != "refs"}, **fields}))


class _Writer:
    """A plain model that answers one text and records what it was sent."""

    def __init__(self, text):
        self.text, self.sent = text, []

    def model_copy(self, update):
        return self

    def invoke(self, messages):
        self.sent.append(messages)
        return AIMessage(content=self.text)


class BrowserStepTests(unittest.TestCase):
    def setUp(self):
        StructuredFakeModel.seen_messages.clear()
        StructuredFakeModel.structured.clear()

    def test_a_step_is_one_canonical_answer_on_the_fixed_browser_model(self):
        fake = StructuredFakeModel(responses=[_answer(kind="read", refs=["p1:0"])])
        self.assertEqual(step(lambda: fake, "openai", REQUEST), {"kind": "read", "refs": ["p1:0"]})
        self.assertEqual(StructuredFakeModel.structured[0][0], ThreadAnswer)
        fake = StructuredFakeModel(responses=[_haiku(kind="click", ref="p1:0")])
        self.assertEqual(step(lambda: fake, "anthropic", REQUEST), {"kind": "click", "ref": "p1:0"})
        self.assertEqual(StructuredFakeModel.structured[1][0], StepAnswer)
        self.assertEqual(
            [model(provider) for provider in ("openai", "anthropic")],
            [BROWSER_MODELS["openai"], BROWSER_MODELS["anthropic"]],
        )
        with self.assertRaises(RuntimeContractError):
            model("gemini")

    def test_each_provider_sends_its_output_cap_and_its_effort(self):
        for provider, effort in (("openai", "medium"), ("anthropic", "low")):
            config = agent_runtime.ProviderConfig(provider=provider, model=model(provider), api_key="secret-test-key")
            client = provider_client.provider_model(config)
            payload = browser_step.capped(client, provider)._get_request_payload([("user", "hi")])
            with self.subTest(provider=provider):
                key = "max_output_tokens" if provider == "openai" else "max_tokens"
                self.assertEqual(payload[key], browser_step.MAX_STEP_OUTPUT_TOKENS)
                sent = payload["reasoning"] if provider == "openai" else payload["output_config"]
                self.assertEqual(sent, {"effort": effort})

    def test_the_thread_renders_as_a_conversation_that_only_grows_at_its_end(self):
        image = base64.b64encode(PNG_SIGNATURE + b"pixels").decode()
        first = validate({**REQUEST, "thread": EMPTY, "screenshot": image})
        rendered = browser_thread.messages(first)
        self.assertIsInstance(rendered[0], SystemMessage)
        self.assertIn("untrusted data", rendered[0].content)
        self.assertEqual(json.loads(rendered[1].content), {"objective": REQUEST["objective"], "language": "pt-BR"})
        self.assertEqual(rendered[-1].content[1], {"type": "image", "base64": image, "mime_type": "image/png"})
        # The next request's history holds that page, its image now a fixed text, then the step taken on it.
        entry = {**ENTRY, "text": REQUEST["observation"], "downloads": REQUEST["downloads"], "screenshot": True}
        later = browser_thread.messages(validate({**REQUEST, "thread": {**EMPTY, "steps": [entry]}}))
        self.assertEqual(later[2].content, rendered[-1].content[0]["text"] + browser_thread.SCREENSHOT_SHOWN)
        self.assertTrue(later[3].content.startswith("Your step: "))
        # Without a screenshot each earlier request is exactly a prefix of the next.
        requests = []
        thread = dict(EMPTY, steps=[])
        for number in range(4):
            observation = f"url: https://example.com/{number}"
            requests.append(
                browser_thread.messages(validate({**REQUEST, "thread": thread, "observation": observation}))
            )
            page = {**ENTRY, "text": observation, "downloads": REQUEST["downloads"]}
            thread = {**thread, "steps": [*thread["steps"], page]}
        for earlier, later in itertools.pairwise(requests):
            self.assertEqual(later[: len(earlier)], earlier)

    def test_folded_steps_and_notes_form_one_earlier_message_of_untrusted_notes(self):
        folded = [{key: ENTRY[key] for key in ("step", "outcome", "page")}]
        thread = {**EMPTY, "folded": folded, "folded_omitted": 3, "notes": ["V101: R$ 9.500"], "notes_omitted": 1}
        rendered = browser_thread.messages(validate({**REQUEST, "thread": thread}))
        earlier = rendered[2].content
        self.assertTrue(earlier.startswith(browser_thread.EARLIER + "(3 earlier steps omitted)\nStep 4 at about:blank"))
        self.assertIn("your own notes of untrusted pages", earlier)
        self.assertTrue(earlier.endswith("(1 earlier notes omitted)\n\nV101: R$ 9.500"))
        self.assertTrue(rendered[-1].content.startswith("Page at step 5:\n"))

    def test_the_rendered_text_stays_within_the_protocols_measure(self):
        long_entry = {
            **ENTRY,
            "text": "é" * 6_000,
            "page": {"url": "https://example.com/" + "a" * 2_000, "title": "t" * 300},
        }
        records = [{key: long_entry[key] for key in ("step", "outcome", "page")}] * 3
        thread = {**EMPTY, "folded": records, "notes": ["n" * 8_000], "steps": [long_entry] * 2}
        request = {**REQUEST, "thread": thread, "observation": "o" * 20_000}
        admitted = validate(request)
        rendered = browser_thread.messages(admitted, browser_thread.FINAL)
        measured = browser_protocol.request_text_bytes(
            admitted["objective"], thread, admitted["observation"], admitted["downloads"]
        )
        self.assertLessEqual(browser_thread.text_bytes(rendered[1:]), measured + len(browser_thread.FINAL.encode()))

    def test_haikus_step_stays_within_the_protocols_measure(self):
        record = {
            "step": {"kind": "wait", "note": "n" * 300},
            "outcome": "ok",
            "page": {"url": "u" * 2_000, "title": "t" * 300},
        }
        lines = browser_protocol.MAX_FOLDED_BYTES // browser_protocol.record_bytes(record)
        entry = {**record, "text": "x" * 1_000, "downloads": None, "screenshot": False}
        thread = {**EMPTY, "folded": [record] * lines, "steps": [entry] * 5}
        admitted = validate({**REQUEST, "thread": thread})
        rendered = browser_thread.text_bytes([_prompt(admitted)[1]])
        measured = browser_protocol.request_text_bytes(
            admitted["objective"], thread, admitted["observation"], admitted["downloads"]
        )
        self.assertLessEqual(rendered, measured)

    def test_haiku_keeps_its_step_on_the_newest_records(self):
        image = base64.b64encode(PNG_SIGNATURE + b"pixels").decode()
        records = [{key: ENTRY[key] for key in ("step", "outcome", "page")}] * 25
        thread = {**EMPTY, "folded": records, "steps": [ENTRY] * 20}
        system, human = _prompt(validate({**REQUEST, "thread": thread, "screenshot": image}))
        self.assertIsInstance(system, SystemMessage)
        self.assertIsInstance(human, HumanMessage)
        task = json.loads(human.content[0]["text"])
        self.assertEqual(set(task), {"objective", "language", "history", "downloads"})
        self.assertEqual(len(task["history"]), browser_step.HAIKU_RECORDS)
        self.assertTrue(human.content[1]["text"].startswith("Current page (untrusted data):\n"))
        self.assertEqual(human.content[2], {"type": "image", "base64": image, "mime_type": "image/png"})

    def test_an_answer_outside_the_closed_set_is_a_response_failure(self):
        for answer in (_answer(kind="click"), _answer(kind="wait", url="https://example.com"), AIMessage(content="no")):
            with self.subTest(answer=answer.content), self.assertRaises(ProviderResponseError):
                step(lambda answer=answer: StructuredFakeModel(responses=[answer]), "openai", REQUEST)

    def test_a_request_past_its_priced_bound_is_refused_before_any_model_call(self):
        def unreachable():
            raise AssertionError("no model call")

        rendered = browser_thread.messages(validate(REQUEST))
        self.assertLessEqual(input_bound(rendered), MAX_PRICED_INPUT_TOKENS)
        with (
            self.assertRaisesRegex(RuntimeContractError, "priced bound"),
            mock.patch.object(browser_step, "MAX_PRICED_INPUT_TOKENS", input_bound(rendered) - 1),
        ):
            step(unreachable, "openai", REQUEST)

    def test_a_malformed_request_is_refused_before_any_model_call(self):
        oversized = base64.b64encode(PNG_SIGNATURE + b"x" * (2 * 1024 * 1024)).decode()
        for change in (
            {"objective": ""},
            {"language": "fr"},
            {"thread": {**EMPTY, "folded": [{**ENTRY, "step": {"kind": "done", "summary": "x"}}]}},
            {"thread": None},
            {"observation": "x" * (24 * 1024 + 1)},
            {"screenshot": "not base64!"},
            {"screenshot": base64.b64encode(b"GIF89a").decode()},
            {"screenshot": oversized},
            {"screenshot": 7},
            {"downloads": {"files": ["b", "a"], "pending": 0, "truncated": False}},
            {"downloads": []},
        ):
            with self.subTest(change=str(change)[:60]), self.assertRaises(RuntimeContractError):
                validate({**REQUEST, **change})


class BrowserFinishingTests(unittest.TestCase):
    def test_a_fold_extends_the_thread_with_its_instruction_and_returns_the_agents_note(self):
        writer = _Writer("Vaga V101: R$ 9.500\x07 por mês")
        answer = fold(lambda: writer, "openai", {**REQUEST, "first": 1, "last": 1})
        self.assertEqual(answer, {"note": "Vaga V101: R$ 9.500  por mês"})
        sent = writer.sent[0]
        self.assertEqual(sent[:-1], browser_thread.messages(validate(REQUEST)))
        self.assertEqual(sent[-1].content, browser_thread.FOLD.format(first=1, last=1))
        self.assertEqual(fold(lambda: _Writer("   "), "openai", {**REQUEST, "first": 1, "last": 1}), {"note": None})
        for numbers in ({"first": 0, "last": 1}, {"first": 2, "last": 1}, {"first": "1", "last": 1}):
            with self.subTest(numbers=numbers), self.assertRaises(RuntimeContractError):
                fold(lambda: writer, "openai", {**REQUEST, **numbers})

    def test_a_final_answer_is_written_from_the_thread_within_the_summary_bound(self):
        writer = _Writer("x" * (browser_protocol.MAX_SUMMARY_BYTES + 50))
        answer = final(lambda: writer, "openai", REQUEST)
        self.assertEqual(len(answer["summary"].encode()), browser_protocol.MAX_SUMMARY_BYTES)
        self.assertEqual(writer.sent[0][-1].content, browser_thread.FINAL)

    def test_the_check_judges_completeness_only(self):
        request = {"objective": REQUEST["objective"], "language": "pt-BR", "summary": "Prazo: 30/11."}
        complete = StructuredFakeModel(responses=[AIMessage(content=json.dumps({"complete": True, "missing": None}))])
        self.assertEqual(check(lambda: complete, "openai", request), {"complete": True, "missing": None})
        short = {"complete": False, "missing": "Falta a fonte."}
        incomplete = StructuredFakeModel(responses=[AIMessage(content=json.dumps(short))])
        self.assertEqual(check(lambda: incomplete, "openai", request), short)
        with self.assertRaises(RuntimeContractError):
            check(lambda: complete, "openai", {**request, "summary": ""})

    def test_a_written_answer_that_fails_or_a_verdict_outside_its_form_is_a_response_failure(self):
        class Broken(_Writer):
            def invoke(self, messages):
                raise OSError("connection reset")

        with self.assertRaises(ProviderResponseError):
            final(lambda: Broken(""), "openai", REQUEST)
        with (
            mock.patch.object(browser_step, "MAX_PRICED_INPUT_TOKENS", 1),
            self.assertRaisesRegex(RuntimeContractError, "priced bound"),
        ):
            final(lambda: _Writer("x"), "openai", REQUEST)
        request = {"objective": REQUEST["objective"], "language": "pt-BR", "summary": "ok"}
        vague = StructuredFakeModel(responses=[AIMessage(content=json.dumps({"complete": False, "missing": None}))])
        with self.assertRaises(ProviderResponseError):
            check(lambda: vague, "openai", request)

    def test_a_request_past_the_request_measure_is_refused(self):
        with (
            mock.patch.object(browser_protocol, "MAX_REQUEST_TEXT_BYTES", 10),
            self.assertRaises(RuntimeContractError),
        ):
            validate(REQUEST)

    def test_haiku_folds_checks_and_finishes_nothing_and_makes_no_call(self):
        def unreachable():
            raise AssertionError("no model call")

        request = {"objective": REQUEST["objective"], "language": "pt-BR", "summary": "ok"}
        self.assertEqual(fold(unreachable, "anthropic", {**REQUEST, "first": 1, "last": 1}), {"note": None})
        self.assertEqual(final(unreachable, "anthropic", REQUEST), {"summary": None})
        self.assertEqual(check(unreachable, "anthropic", request), {"complete": None, "missing": None})


if __name__ == "__main__":
    unittest.main()
