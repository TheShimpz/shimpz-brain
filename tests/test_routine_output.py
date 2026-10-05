"""What each Routine run does with its result, chosen by the person's said words (ADR-0092 amendment, 2026-10-05)."""

from __future__ import annotations

import unittest

from test_routine import CONTRACTS, LISTED, _compiled, _said, _source

import routine


class OutputTests(unittest.TestCase):
    """What each run does with its result is chosen by the user's own said words (ADR-0092 amendment, 2026-10-05)."""

    SAID = "Toda segunda às 9h, diga olá para Ana e me mostre o resultado só quando mudar"

    def wire(self, output: routine.Output | None, steps: list[routine.Step] | None = None, target=None):
        compiled = _compiled(output=output, **({"steps": steps} if steps else {}))
        return routine.change(compiled, _said(self.SAID), CONTRACTS, target)

    def test_said_words_choose_a_closed_disposition(self) -> None:
        changes = routine.Output(mode="changes", step="greet", instruction="me mostre o resultado só quando mudar")
        self.assertEqual(
            self.wire(changes)["output"],
            {"mode": "changes", "step": "greet", "instruction": "me mostre o resultado só quando mudar"},
        )
        none = routine.Output(mode="none", step=None, instruction="diga olá")
        self.assertEqual(self.wire(none)["output"], {"mode": "none", "step": None, "instruction": "diga olá"})
        kept = routine.Output(mode="kept", step=None, instruction=None)
        self.assertEqual(self.wire(kept, target=LISTED)["output"], {"mode": "kept"})

    def test_a_disposition_no_said_words_choose_or_that_names_no_shown_step_is_unproven(self) -> None:
        for output, target in (
            (None, None),
            (routine.Output(mode="kept", step=None, instruction=None), None),
            (routine.Output(mode="show", step="greet", instruction="mostre tudo"), None),
            (routine.Output(mode="show", step="other", instruction="me mostre"), None),
            (routine.Output(mode="none", step="greet", instruction="diga"), None),
            (routine.Output(mode="chain", step=None, instruction="diga"), None),
        ):
            with self.subTest(output=output), self.assertRaises(routine.UnprovenError):
                self.wire(output, target=target)
        # Words only an earlier send holds never choose it.
        cited = routine.Output(mode="show", step="greet", instruction="mostre o resultado")
        with self.assertRaises(routine.UnprovenError):
            routine.change(
                _compiled(output=cited),
                _said("Toda segunda às 9h, diga olá para Ana", ("mostre o resultado",)),
                CONTRACTS,
                None,
            )

    def test_handing_the_result_on_as_text_chains_two_steps(self) -> None:
        handed = _source(
            "name", "step_text", value_json=None, origins=[], step="greet", pointer="", instruction="diga olá para Ana"
        )
        steps = [
            routine.Step(id="greet", assistant="hello-pulse", action="hello", inputs=[_source()]),
            routine.Step(id="again", assistant="hello-pulse", action="hello", inputs=[handed]),
        ]
        chain = routine.Output(mode="chain", step=None, instruction="diga olá para Ana")
        wire = self.wire(chain, steps)
        self.assertEqual(wire["output"]["mode"], "chain")
        self.assertEqual(
            wire["steps"][1]["input"]["name"],
            {"kind": "step_text", "step": "greet", "pointer": "", "instruction": "diga olá para Ana"},
        )
        # Words only an earlier send holds never relate two steps.
        cited = _source(
            "name", "step_text", value_json=None, origins=[], step="greet", pointer="", instruction="passe adiante"
        )
        relayed = [steps[0], routine.Step(id="again", assistant="hello-pulse", action="hello", inputs=[cited])]
        with self.assertRaises(routine.UnprovenError):
            routine.change(
                _compiled(output=chain, steps=relayed), _said(self.SAID, ("passe adiante",)), CONTRACTS, None
            )
        unrelated = _source("name", "step_text", value_json=None, origins=[], step="greet", pointer="", instruction="x")
        steps[1] = routine.Step(id="again", assistant="hello-pulse", action="hello", inputs=[unrelated])
        with self.assertRaises(routine.UnprovenError):
            self.wire(chain, steps)
