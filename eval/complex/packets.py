"""Blinded paired adjudication packets of precision-v3 episodes for the owner (ADR-0094).

One packet per template: one episode instance (language and repetition) that both arms of a campaign completed,
chosen by a seeded hash, with its two trajectories shown as X and Y in a seeded order. A packet carries what the owner
needs to judge communication and to check the state: the episode's language, every user message, each turn's
obligation, each trajectory's replies and Action records, and each trajectory's final-state differences from the
reference workflow with the oracle's counts. It carries no provider, model, arm, repetition, campaign, cost, or judge
verdict; the key that maps X and Y back to arms is written apart, so labels are collected blind. The labels file has
one empty slot per trajectory and criterion of ``eval.complex.judging`` and records how they were collected.

Experiment-only; ``python -m eval.complex.packets --transcript T --seed S --out DIR`` writes ``packets.json``,
``key.json``, and ``labels.json`` owner-only into a private directory outside every repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from eval import private
from eval.complex import judging
from eval.complex.model import CORPUS_ID, reference_states
from eval.complex.templates import TEMPLATES, TEMPLATES_BY_ID
from eval.corpus import LANGUAGE_NAMES


def _rank(seed: str, *parts: object) -> str:
    return hashlib.sha256(json.dumps([seed, *parts]).encode()).hexdigest()


def differences(template_id: str, final: Mapping[str, object]) -> list[dict[str, object]]:
    """Each state key whose final value differs from the reference workflow's, with both values."""
    expected = reference_states(TEMPLATES_BY_ID[template_id])[-1]
    return [
        {"key": key, "actual": final.get(key), "expected": expected.get(key)}
        for key in sorted(set(final) | set(expected))
        if final.get(key) != expected.get(key)
    ]


def _trajectory(record: Mapping[str, object]) -> dict[str, object]:
    turns = [
        {
            "user_message": turn["user"],
            "assistant_reply": turn["reply"],
            "action_record": [
                {key: entry[key] for key in ("assistant", "action", "input", "result", "failed") if key in entry}
                for entry in turn["record"]
            ],
        }
        for turn in record["turns"]
    ]
    final = record["checkpoints"][-1] if record["checkpoints"] else {}
    return {
        "turns": turns,
        "state_differences": differences(str(record["template"]), final),
        "oracle": dict(record["oracle"]),
    }


def build(records: Iterable[Mapping[str, object]], seed: str) -> tuple[list[dict], dict[str, dict], dict]:
    """The packets, the private key from packet trajectory to arm, and the empty labels file."""
    instances: dict[tuple[str, str, int], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for record in records:
        if record["status"] == "completed":
            instances[str(record["campaign"]), str(record["episode"]), int(record["repetition"])][
                str(record["arm"])
            ] = record
    by_template: dict[str, list[tuple[tuple[str, str, int], dict[str, Mapping[str, object]]]]] = defaultdict(list)
    for key, arms in instances.items():
        if len(arms) == 2:
            by_template[str(next(iter(arms.values()))["template"])].append((key, arms))
    packets, keys = [], {}
    for template in TEMPLATES:
        candidates = sorted(by_template.get(template.id, ()), key=lambda item: _rank(seed, *item[0]))
        if not candidates:
            continue
        (campaign, episode, repetition), arms = candidates[0]
        order = sorted(arms, key=lambda arm: _rank(seed, episode, repetition, arm))
        packet_id = f"p{len(packets) + 1:02d}"
        first = arms[order[0]]
        packets.append(
            {
                "id": packet_id,
                "stratum": template.stratum,
                "expected_reply_language": LANGUAGE_NAMES[str(first["locale"])],
                "obligations": [step.communicate for step in template.steps],
                "trajectories": {label: _trajectory(arms[arm]) for label, arm in zip("XY", order, strict=True)},
            }
        )
        keys[packet_id] = {
            "campaign": campaign,
            "episode": episode,
            "repetition": repetition,
            "X": order[0],
            "Y": order[1],
        }
    labels = {
        "corpus": CORPUS_ID,
        "judge_version": judging.VERSION,
        "collection": {"blind_to_judge_verdicts": None, "frozen_judge_identity": None},
        "labels": {
            packet["id"]: {label: dict.fromkeys(judging.CRITERIA) for label in packet["trajectories"]}
            for packet in packets
        },
    }
    return packets, keys, labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        records = [json.loads(line) for line in args.transcript.read_text(encoding="utf-8").splitlines() if line]
        packets, keys, labels = build(records, args.seed)
        args.out.mkdir(mode=0o700, exist_ok=True)
        for name, body in (("packets.json", packets), ("key.json", keys), ("labels.json", labels)):
            private.write_private(args.out / name, json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True))
    except OSError, ValueError, KeyError, TypeError:
        print("packet generation failed", file=sys.stderr)
        return 2
    print(json.dumps({"packets": len(packets), "templates": len(TEMPLATES)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
