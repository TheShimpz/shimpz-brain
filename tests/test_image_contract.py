from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UV_IMAGE = "ghcr.io/astral-sh/uv:0.12.1@sha256:cf4eedcaa81655197f625739489effcbe71b61ceb1506f332c3facae5deceded"


class StaticBrainImageContractTests(unittest.TestCase):
    def test_static_builder_obtains_content_addressed_uv_without_apt(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn(f"FROM {UV_IMAGE} AS uv", dockerfile)
        self.assertIn("--mount=type=bind,from=uv,source=/uv,target=/tmp/uv", dockerfile)
        self.assertNotIn("uv-install.sh", dockerfile)
        self.assertNotIn("apt-get", dockerfile)
        self.assertNotIn("curl", dockerfile)

    def test_static_runtime_derives_from_an_epoch_free_dependency_layer(self):
        # Shimpz ADR-0098: no commit-time input reaches the dependency layer, so an unchanged lock reuses its bytes.
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("\nFROM dependencies AS runtime\n", dockerfile)
        stages = dict(re.findall(r"(?ms)^FROM \S+ AS (\w+)\n(.*?)(?=^FROM |\Z)", dockerfile))
        self.assertEqual(["dependencies", "runtime", "uv"], sorted(stages))
        for stage in ("uv", "dependencies"):
            with self.subTest(stage=stage):
                self.assertNotRegex(stages[stage], r"(?m)^(ARG SOURCE_DATE_EPOCH|WORKDIR|COPY|ADD)\b")
        dependencies = re.sub(r"\\\n\s*", " ", stages["dependencies"])
        for mount in (
            "--mount=type=tmpfs,target=/tmp",
            "--mount=type=bind,source=pyproject.toml,target=/tmp/project/pyproject.toml",
            "--mount=type=bind,source=uv.lock,target=/tmp/project/uv.lock",
        ):
            self.assertIn(mount, dependencies)
        self.assertIn("uv sync --frozen --no-install-project --no-dev --python 3.14", dependencies)
        self.assertIn("compileall -q -f --invalidation-mode checked-hash /opt/venv", dependencies)
        self.assertTrue(dependencies.rstrip().endswith("find /opt -depth -exec touch -h -d @0 {} +"))

    def test_static_image_runs_only_the_non_root_http_runtime(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("FROM python:3.14-slim@sha256:", dockerfile)
        self.assertIn("USER brainruntime", dockerfile)
        self.assertIn("HEALTHCHECK --interval=10s", dockerfile)
        self.assertIn("socket.create_connection", dockerfile)
        self.assertIn("GET /health HTTP/1.0", dockerfile)
        self.assertIn("HTTP/1.1 200 OK", dockerfile)
        self.assertNotIn("urllib.request", dockerfile)
        self.assertIn('"runtime_api:app"', dockerfile)
        self.assertIn('"--workers", "1"', dockerfile)
        self.assertIn('"--no-access-log"', dockerfile)
        self.assertNotIn("COPY rootfs", dockerfile)
        self.assertNotIn("COPY codex", dockerfile)

    def test_profile_owns_runtime_paths_while_image_owns_tracing_and_allocator_defaults(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("SHIMPZ_BRAIN_RUNTIME_TOKEN_GID=10016", dockerfile)
        self.assertNotIn("SHIMPZ_BRAIN_RUNTIME_TOKEN_FILE=", dockerfile)
        self.assertNotIn("SHIMPZ_BRAIN_RUNTIME_STATE=", dockerfile)
        self.assertIn("LANGSMITH_TRACING=false", dockerfile)
        self.assertIn("MALLOC_ARENA_MAX=2", dockerfile)
        self.assertIn("MALLOC_MMAP_THRESHOLD_=131072", dockerfile)

    def test_runtime_artifact_excludes_the_independent_egress_role(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn(
            "action_labels.py action_purpose.py action_schema.py agent_runtime.py attachments.py capability_plan.py "
            "clarification.py context_budget.py action_tool.py intent_fast_path.py "
            "intent_route.py interface_language.py memory.py model_usage.py provider_cancel.py provider_client.py "
            "routine.py "
            "routine_recovery.py routine_words.py runtime_api.py runtime_errors.py structured.py tool_refusal.py "
            "turn_pins.py turn_prompt.py \\\n"
            "    model_catalog.json /app/",
            dockerfile,
        )
        self.assertIn(
            "COPY --chown=brainruntime:brainruntime protocol/team/action/v1/schema.py /app/protocol/team/action/v1/\n",
            dockerfile,
        )
        self.assertIn(
            "COPY --chown=brainruntime:brainruntime protocol/team/http/v1/identifiers.py "
            "protocol/team/http/v1/purpose.py \\\n    protocol/team/http/v1/turn.py /app/protocol/team/http/v1/\n",
            dockerfile,
        )
        self.assertNotIn("egress/", dockerfile)
        self.assertNotIn("/var/log/brain-egress", dockerfile)
        self.assertNotIn("SHIMPZ_EGRESS_ALLOW", dockerfile)


if __name__ == "__main__":
    unittest.main()
