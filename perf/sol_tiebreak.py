"""Run ``eval.precision`` with gpt-6.1-sol as the tiebreak judge in place of claude-sonnet-5-5 (ADR-0094).

A declared deviation for when the Anthropic test key has no credit: the gpt-6-luna primary judge is unchanged, the
tiebreak model differs from the calibrated judge, so the judge identity changes and every report built from these
verdicts is exploratory. Both judges read ``--key-file``. From the Brain checkout:

    PYTHONPATH=. uv run --frozen --python 3.14 python perf/sol_tiebreak.py judge --transcript <arms.jsonl>
        --key-file ../.gpt-key --tiebreak-key-file ../.gpt-key --cap 3.0 --out <judged.jsonl>   (one command line)
"""

import argparse
import sys

from eval import cost as eval_cost
from eval import judge, precision

MODELS = ("gpt-6-luna", "gpt-6.1-sol")


def _judges(args: argparse.Namespace, spend: list[eval_cost.Cost]) -> tuple[precision.Judgment, precision.Judgment]:
    from eval.intent_route import _key
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    key = _key(args.key_file)
    judgments = []
    for model_id in MODELS:
        model = ChatOpenAI(
            model=model_id,
            api_key=SecretStr(key),
            max_tokens=judge.MAX_OUTPUT_TOKENS,
            max_retries=0,
            timeout=judge.TIMEOUT_SECONDS,
            use_responses_api=True,
            reasoning_effort="low",
        )
        judgments.append(
            precision.metered(model_id, lambda item, model=model: judge.judge(model, "openai", item), spend)
        )
    return judgments[0], judgments[1]


def main() -> int:
    judge.JUDGE_MODELS = {"openai": MODELS[0], "openai-tiebreak": MODELS[1]}
    precision._judges = _judges
    return precision.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
