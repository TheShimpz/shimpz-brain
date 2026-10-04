"""Provider-free checks for the blinded precision-v3 adjudication packets (ADR-0094)."""

from __future__ import annotations

import io
import json
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from eval.complex import judging, model, packets
from eval.complex.templates import TEMPLATES


def _record(template, arm: str, repetition: int = 0, *, status: str = "completed", campaign: str = "c") -> dict:
    world, checkpoints = model.run_reference(template)
    if arm == "bravo":
        checkpoints[-1] = {**checkpoints[-1], "extra-key": "changed"}
    return {
        "campaign": campaign,
        "provider": "openai",
        "model": "gpt-6-luna",
        "arm": arm,
        "repetition": repetition,
        "episode": f"{template.id}.en",
        "template": template.id,
        "locale": "en",
        "status": status,
        "turns": [
            {"step": index, "user": f"message {index}", "reply": "Done.", "record": [dict(e) for e in world.ledger[:1]]}
            for index in range(template.turns)
        ],
        "checkpoints": checkpoints,
        "oracle": {"passed": arm == "alpha"},
        "usd": 0.01,
    }


class PacketTests(unittest.TestCase):
    def test_one_blinded_packet_per_template_with_a_private_key(self):
        records = [_record(t, arm, rep) for t in TEMPLATES[:6] for arm in ("alpha", "bravo") for rep in (0, 1)]
        records.append(_record(TEMPLATES[6], "alpha"))  # only one arm: no packet
        records += [_record(TEMPLATES[7], arm, status="budget-stopped") for arm in ("alpha", "bravo")]
        built, keys, labels = packets.build(records, "seed")
        self.assertEqual([p["stratum"] for p in built], [t.stratum for t in TEMPLATES[:6]])
        text = json.dumps(built)
        for leaked in ("openai", "luna", "alpha", "bravo", "repetition", "campaign", "usd", "gpt-6"):
            self.assertNotIn(leaked, text)
        orders = set()
        for packet in built:
            key = keys[packet["id"]]
            orders.add(key["X"])
            for label in ("X", "Y"):
                trajectory = packet["trajectories"][label]
                self.assertEqual(trajectory["oracle"]["passed"], key[label] == "alpha")
                self.assertEqual(bool(trajectory["state_differences"]), key[label] == "bravo")
            self.assertEqual(len(packet["obligations"]), len(packet["trajectories"]["X"]["turns"]))
        self.assertEqual(orders, {"alpha", "bravo"})
        self.assertEqual(packets.build(records, "seed"), (built, keys, labels))
        self.assertEqual(set(labels["labels"]), set(keys))
        slots = labels["labels"][built[0]["id"]]
        self.assertEqual(slots, {"X": dict.fromkeys(judging.CRITERIA), "Y": dict.fromkeys(judging.CRITERIA)})
        self.assertIsNone(labels["collection"]["blind_to_judge_verdicts"])
        changed = packets.differences(TEMPLATES[0].id, {"extra-key": 1, **model.reference_states(TEMPLATES[0])[-1]})
        self.assertEqual(changed, [{"key": "extra-key", "actual": 1, "expected": None}])
        empty = {**_record(TEMPLATES[0], "alpha"), "checkpoints": []}
        self.assertIsInstance(packets._trajectory(empty)["state_differences"], list)

    def test_the_command_writes_owner_only_files_and_fails_closed(self):
        records = [_record(t, arm) for t in TEMPLATES[:2] for arm in ("alpha", "bravo")]
        with tempfile.TemporaryDirectory() as directory:
            transcript = Path(directory, "t.jsonl")
            transcript.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            out = Path(directory, "packets")
            # Generation is called directly on valid input, so a failure keeps its traceback.
            self.assertEqual(packets.generate(transcript, "s", out), 2)
            for name in ("packets.json", "key.json", "labels.json"):
                self.assertEqual(Path(out, name).stat().st_mode & 0o077, 0)
            argv = ["--transcript", str(transcript), "--seed", "s", "--out", str(out)]
            with (
                mock.patch.object(packets, "generate", return_value=2) as generate,
                redirect_stdout(io.StringIO()) as printed,
            ):
                self.assertEqual(packets.main(argv), 0)
            generate.assert_called_once_with(transcript, "s", out)
            self.assertEqual(json.loads(printed.getvalue()), {"packets": 2, "templates": len(TEMPLATES)})
            transcript.write_text("{not json\n", encoding="utf-8")
            with redirect_stderr(io.StringIO()):
                self.assertEqual(packets.main(argv), 2)
            with (
                mock.patch.object(sys, "argv", ["packets", *argv]),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                runpy.run_module("eval.complex.packets", run_name="__main__")


if __name__ == "__main__":
    unittest.main()
