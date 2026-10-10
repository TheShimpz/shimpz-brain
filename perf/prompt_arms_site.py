"""The ``sitecustomize`` of one disposable precision Brain process: the umbrella's ceiling hook, then one prompt arm.

``perf/precision_arms.py`` copies this file in place of ``.tests/perf/precision_brain_budget.py``, names that hook in
``PROMPT_ARMS_BUDGET_HOOK`` and the arm in ``PROMPT_ARMS_ARM``; no product image or process loads it. The hook puts
the Brain checkout on the import path, so the arm comes from ``perf/prompt_arms.py`` of the Brain it serves.
"""

import importlib
import os
import runpy

if os.environ.get("SHIMPZ_PRECISION_BUDGET_FILE"):
    runpy.run_path(os.environ["PROMPT_ARMS_BUDGET_HOOK"])
    importlib.import_module("perf.prompt_arms").install(os.environ["PROMPT_ARMS_ARM"])
