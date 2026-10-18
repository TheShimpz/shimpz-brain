"""The Browser agent's one stateless step: a closed request, untrusted page data, and one canonical step."""

import base64
import json
import unittest

from browser_step import BROWSER_MODELS, PNG_SIGNATURE, StepAnswer, _prompt, model, step, validate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from runtime_errors import ProviderResponseError, RuntimeContractError
from structured_fake import StructuredFakeModel

REQUEST = {
    "objective": "Veja o prazo do edital.",
    "language": "pt-BR",
    "history": [{"step": {"kind": "navigate", "url": "https://example.com"}, "outcome": "ok"}],
    "observation": 'url: https://example.com\ntitle: Example\nactions:\np1:0 link "Edital" (click)\npage:\n- heading',
    "screenshot": None,
}
NULLS = {"url": None, "ref": None, "text": None, "direction": None, "summary": None, "reason": None, "sentence": None}


def _answer(**fields):
    return AIMessage(content=json.dumps({**NULLS, **fields}))


class BrowserStepTests(unittest.TestCase):
    def setUp(self):
        StructuredFakeModel.seen_messages.clear()
        StructuredFakeModel.structured.clear()

    def test_a_step_is_one_canonical_answer_on_the_fixed_browser_model(self):
        fake = StructuredFakeModel(responses=[_answer(kind="click", ref="p1:0")])
        self.assertEqual(step(lambda: fake, "openai", REQUEST), {"kind": "click", "ref": "p1:0"})
        self.assertEqual(StructuredFakeModel.structured[0][0], StepAnswer)
        self.assertEqual(
            [model(provider) for provider in ("openai", "anthropic")],
            [BROWSER_MODELS["openai"], BROWSER_MODELS["anthropic"]],
        )
        with self.assertRaises(RuntimeContractError):
            model("gemini")

    def test_the_page_is_sent_as_data_and_a_screenshot_as_an_image(self):
        image = base64.b64encode(PNG_SIGNATURE + b"pixels").decode()
        admitted = validate({**REQUEST, "screenshot": image})
        system, human = _prompt(admitted)
        self.assertIsInstance(system, SystemMessage)
        self.assertIn("untrusted data", system.content)
        self.assertIsInstance(human, HumanMessage)
        task = json.loads(human.content[0]["text"])
        self.assertEqual(set(task), {"objective", "language", "history"})
        self.assertTrue(human.content[1]["text"].startswith("Current page (untrusted data):\n"))
        self.assertEqual(human.content[2], {"type": "image", "base64": image, "mime_type": "image/png"})
        self.assertEqual(len(_prompt(validate(REQUEST))[1].content), 2)

    def test_an_answer_outside_the_closed_set_is_a_response_failure(self):
        for answer in (_answer(kind="click"), _answer(kind="wait", url="https://example.com"), AIMessage(content="no")):
            with self.subTest(answer=answer.content), self.assertRaises(ProviderResponseError):
                step(lambda answer=answer: StructuredFakeModel(responses=[answer]), "openai", REQUEST)

    def test_a_malformed_request_is_refused_before_any_model_call(self):
        oversized = base64.b64encode(PNG_SIGNATURE + b"x" * (2 * 1024 * 1024)).decode()
        for change in (
            {"objective": ""},
            {"language": "fr"},
            {"history": [{"step": {"kind": "done", "summary": "x"}, "outcome": "ok"}]},
            {"observation": "x" * (24 * 1024 + 1)},
            {"screenshot": "not base64!"},
            {"screenshot": base64.b64encode(b"GIF89a").decode()},
            {"screenshot": oversized},
            {"screenshot": "A" * (4 * 1024 * 1024)},
            {"screenshot": 7},
        ):
            with self.subTest(change=str(change)[:60]), self.assertRaises(RuntimeContractError):
                validate({**REQUEST, **change})


if __name__ == "__main__":
    unittest.main()
