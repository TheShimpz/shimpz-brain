"""Arrange-and-act helpers shared by the fresh evaluation strata tests; every assertion stays in its own test."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from types import ModuleType
from typing import Any

from eval import corpus

Workflow = Callable[[Any], None]
# The oracle result of a reference workflow: passed, with no defect of any kind.
PASSED = {"passed": True, "missing": 0, "wrong": 0, "forbidden": 0, "wrong_scope": 0, "duplicates": 0}


def one(items: list) -> dict:
    (item,) = items
    return item


def call(assistant: str, action: str, arguments: dict) -> Workflow:
    """A workflow that invokes exactly one Action."""

    def workflow(world: Any) -> None:
        world.invoke(assistant, action, arguments)

    return workflow


def run(stratum: ModuleType, *workflows: Workflow) -> Any:
    """A fresh World of ``stratum`` after running every workflow on it in order."""
    world = stratum.FreshWorld()
    for workflow in workflows:
        workflow(world)
    return world


def oracle(stratum: ModuleType, scenario_id: str, world: Any) -> corpus.Oracle:
    return corpus.oracle(stratum.SCENARIOS_BY_ID[scenario_id], world, stratum.INITIAL)


def template_defects(stratum: ModuleType) -> list[tuple[corpus.Template, ...]]:
    """Template sets of ``stratum`` that each carry one structural defect validation must refuse."""
    template, *rest = stratum.TEMPLATES
    return [
        (*stratum.TEMPLATES, template),
        *(
            (dataclasses.replace(template, **change), *rest)
            for change in (
                {"behavior": "guess"},
                {"needed": ("fitness",)},
                {"min_rounds": 9},
                {"expect_clarification": True},
                {"changes": {}},
                {"reference": ""},
                {"messages": {"en": "x"}},
            )
        ),
    ]
