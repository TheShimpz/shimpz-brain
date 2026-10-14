"""The Jev fast-path eval's gate: failed exchanges never pass as evidence (ADR-0077)."""

import unittest

from eval import intent_fast_path as fast_path_eval


class _Answering:
    """Exchanges that always answer the same thing, or never answer."""

    def __init__(self, answers: dict[str, tuple[str, float]] | None) -> None:
        self.answers = answers

    def ask(self, *_args: object) -> dict[str, tuple[str, float]] | None:
        return self.answers


SAFE = {"intent": ("unresolved", 0.99), "capability": ("capability", 0.99)}


class EvalGateTests(unittest.TestCase):
    def test_a_run_without_answers_is_inconclusive_and_fails_its_gate(self) -> None:
        intent = fast_path_eval._intent_run(_Answering(None), ())
        self.assertFalse(intent["conclusive"])
        self.assertFalse(intent["gate_passed"])
        capability = fast_path_eval._capability_run(_Answering(None), "owner", 1)
        self.assertFalse(capability["conclusive"])
        self.assertFalse(capability["gate_passed"])

    def test_a_run_whose_every_exchange_answered_safely_passes(self) -> None:
        intent = fast_path_eval._intent_run(_Answering(SAFE), ())
        self.assertTrue(intent["conclusive"])
        self.assertTrue(intent["gate_passed"])
        capability = fast_path_eval._capability_run(_Answering(SAFE), "owner", 1)
        self.assertTrue(capability["gate_passed"])

    def test_coverage_counts_a_failed_ordinary_case_as_not_covered(self) -> None:
        ordinary = [case for case in fast_path_eval.CASES if case.intent == "ordinary-task"][:2]
        answered = [(ordinary[0], "ordinary-task", 0.99)]
        threshold = str(fast_path_eval.intent_fast_path.CONFIDENCE_THRESHOLD)
        self.assertEqual(fast_path_eval.metrics(answered)[threshold]["ordinary_coverage"], 1.0)
        self.assertEqual(fast_path_eval.metrics(answered, tuple(ordinary))[threshold]["ordinary_coverage"], 0.5)

    def test_the_failure_bound_admits_a_rare_failure_only(self) -> None:
        self.assertTrue(fast_path_eval._conclusive(0, 100))
        self.assertTrue(fast_path_eval._conclusive(2, 100))
        self.assertFalse(fast_path_eval._conclusive(3, 100))
        self.assertFalse(fast_path_eval._conclusive(0, 0))


if __name__ == "__main__":
    unittest.main()
