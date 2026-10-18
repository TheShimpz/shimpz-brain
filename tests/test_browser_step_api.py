"""The Brain's `browser-step` endpoint: authenticated, closed, in its own lane, and metered."""

import unittest

from test_runtime_api import SECRET, TOKEN, FakeRuntime, client

PAYLOAD = {
    "provider": {"provider": "openai", "model": "gpt-6.1-sol", "api_key": SECRET},
    "objective": "Veja o prazo do edital.",
    "language": "pt-BR",
    "history": [],
    "observation": "url: https://example.com",
    "screenshot": None,
}


class BrowserRuntime(FakeRuntime):
    def browser_step(self, provider, request):
        self.seen = (provider, request)
        return {"kind": "wait"}, "gpt-6-luna"


class BrowserStepApiTests(unittest.TestCase):
    def test_a_step_answers_its_step_model_and_usage(self):
        runtime = BrowserRuntime()
        response = client(runtime).post("/v1/browser-step", json=PAYLOAD, headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual((body["step"], body["model"]), ({"kind": "wait"}, "gpt-6-luna"))
        self.assertIn("usage", body)
        provider, request = runtime.seen
        self.assertEqual((provider.provider, request["objective"]), ("openai", PAYLOAD["objective"]))
        self.assertNotIn(SECRET, response.text)

    def test_an_unauthenticated_or_open_request_is_refused(self):
        app = client(BrowserRuntime())
        self.assertEqual(app.post("/v1/browser-step", json=PAYLOAD).status_code, 401)
        headers = {"Authorization": f"Bearer {TOKEN}"}
        for change in ({"language": "fr"}, {"extra": 1}, {"history": [{}] * 41}):
            with self.subTest(change=change):
                self.assertEqual(
                    app.post("/v1/browser-step", json={**PAYLOAD, **change}, headers=headers).status_code, 422
                )

    def test_a_full_lane_answers_at_once(self):
        app = client(BrowserRuntime())
        slots = app.app.state.browser_step_slots
        held = [slots.acquire(blocking=False) for _ in range(2)]
        try:
            response = app.post("/v1/browser-step", json=PAYLOAD, headers={"Authorization": f"Bearer {TOKEN}"})
            self.assertEqual(response.status_code, 429)
        finally:
            for taken in held:
                if taken:
                    slots.release()


if __name__ == "__main__":
    unittest.main()
