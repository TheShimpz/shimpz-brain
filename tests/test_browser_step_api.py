"""The Brain's Browser endpoints: authenticated, closed, in their own lane, and metered."""

import unittest

from test_runtime_api import SECRET, TOKEN, FakeRuntime, client

PAYLOAD = {
    "provider": {"provider": "openai", "model": "gpt-6.1-sol", "api_key": SECRET},
    "objective": "Veja o prazo do edital.",
    "language": "pt-BR",
    "thread": {"folded": [], "folded_omitted": 0, "memory": "", "unnoted": 0, "steps": []},
    "observation": "url: https://example.com",
    "screenshot": None,
    "downloads": {"files": ["report.csv"], "pending": 1, "truncated": False},
}


ANSWERS = {
    "browser-step": {"kind": "wait"},
    "browser-fold": {"memory": "V101: R$ 9.500"},
    "browser-check": {"complete": True, "missing": None},
    "browser-final": {"summary": "ok"},
}


class BrowserRuntime(FakeRuntime):
    def browser_call(self, provider, operation, request):
        self.seen = (provider, operation, request)
        return ANSWERS[operation], "gpt-6-luna"


class BrowserStepApiTests(unittest.TestCase):
    def test_a_step_answers_its_step_model_and_usage(self):
        runtime = BrowserRuntime()
        response = client(runtime).post("/v1/browser-step", json=PAYLOAD, headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual((body["step"], body["model"]), ({"kind": "wait"}, "gpt-6-luna"))
        self.assertIn("usage", body)
        provider, operation, request = runtime.seen
        self.assertEqual((provider.provider, operation), ("openai", "browser-step"))
        self.assertEqual(request["objective"], PAYLOAD["objective"])
        self.assertNotIn(SECRET, response.text)

    def test_each_finishing_operation_answers_its_own_fields(self):
        headers = {"Authorization": f"Bearer {TOKEN}"}
        check = {key: PAYLOAD[key] for key in ("provider", "objective", "language")} | {"summary": "ok"}
        for operation, body in (
            ("browser-fold", {**PAYLOAD, "first": 1, "last": 2}),
            ("browser-check", check),
            ("browser-final", PAYLOAD),
        ):
            with self.subTest(operation=operation):
                runtime = BrowserRuntime()
                response = client(runtime).post(f"/v1/{operation}", json=body, headers=headers)
                self.assertEqual(response.status_code, 200)
                answer = response.json()
                self.assertEqual({key: answer[key] for key in ANSWERS[operation]}, ANSWERS[operation])
                self.assertEqual((runtime.seen[1], answer["model"]), (operation, "gpt-6-luna"))
                self.assertIn("usage", answer)
        app = client(BrowserRuntime())
        self.assertEqual(app.post("/v1/browser-fold", json=PAYLOAD, headers=headers).status_code, 422)
        self.assertEqual(app.post("/v1/browser-check", json={**check, "thread": {}}, headers=headers).status_code, 422)

    def test_an_unauthenticated_or_open_request_is_refused(self):
        app = client(BrowserRuntime())
        self.assertEqual(app.post("/v1/browser-step", json=PAYLOAD).status_code, 401)
        headers = {"Authorization": f"Bearer {TOKEN}"}
        for change in ({"language": "fr"}, {"extra": 1}, {"history": []}, {"thread": None}):
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
