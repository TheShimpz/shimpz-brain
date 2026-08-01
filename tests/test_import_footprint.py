from __future__ import annotations

import multiprocessing
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFERRED_AT_IDLE = (
    "langchain.agents",
    "langchain_anthropic",
    "langchain_openai",
    "langgraph.types",
)
TRACKED_MODULES = (
    *DEFERRED_AT_IDLE,
    "agent_runtime",
    "fastapi",
    "httpx",
    "langchain_core.language_models",
    "langchain_core.tools",
    "langgraph.checkpoint.sqlite",
    "runtime_api",
)


def _loaded_modules(probe: str, connection) -> None:
    preserved_environment = {name: os.environ[name] for name in ("LANG", "PATH") if name in os.environ}
    os.environ.clear()
    os.environ.update(preserved_environment)
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    if probe == "idle":
        import runtime_api

        assert runtime_api.app is not None
    else:
        import agent_runtime

        provider, model = {
            "anthropic": ("anthropic", "claude-sonnet-5"),
            "openai": ("openai", "gpt-5.6-terra"),
        }[probe]
        agent_runtime.provider_model(
            agent_runtime.ProviderConfig(
                provider=provider,
                model=model,
                api_key="synthetic-import-probe-key",
            )
        )
    modules = tuple(sys.modules)
    connection.send({prefix: _has_module(modules, prefix) for prefix in TRACKED_MODULES})
    connection.close()


def _has_module(modules: tuple[str, ...], prefix: str) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for name in modules)


class RuntimeImportFootprintTests(unittest.TestCase):
    def _probe(self, kind: str) -> dict[str, bool]:
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_loaded_modules, args=(kind, sender))
        process.start()
        sender.close()
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            self.fail(f"{kind} import probe timed out")
        self.assertEqual(process.exitcode, 0)
        self.assertTrue(receiver.poll(), f"{kind} import probe returned no module snapshot")
        module_states = receiver.recv()
        receiver.close()
        return module_states

    def test_runtime_api_import_defers_agent_and_provider_stacks(self) -> None:
        module_states = self._probe("idle")

        for expected in (
            "agent_runtime",
            "fastapi",
            "httpx",
            "langchain_core.language_models",
            "langchain_core.tools",
            "langgraph.checkpoint.sqlite",
            "runtime_api",
        ):
            with self.subTest(expected=expected):
                self.assertTrue(module_states[expected])
        for deferred in DEFERRED_AT_IDLE:
            with self.subTest(deferred=deferred):
                self.assertFalse(module_states[deferred])

    def test_each_provider_import_keeps_the_other_adapter_deferred(self) -> None:
        for provider, other in (("openai", "anthropic"), ("anthropic", "openai")):
            with self.subTest(provider=provider):
                module_states = self._probe(provider)

                self.assertTrue(module_states[f"langchain_{provider}"])
                self.assertFalse(module_states[f"langchain_{other}"])
                self.assertFalse(module_states["langchain.agents"])


if __name__ == "__main__":
    unittest.main()
