from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UV_IMAGE = "ghcr.io/astral-sh/uv:0.11.25@sha256:1e3808aa9023d0980e7c15b1fa7c1ac16ff35925780cf5c459858b2d693f01a9"


class StaticBrainImageContractTests(unittest.TestCase):
    def test_static_builder_obtains_content_addressed_uv_without_apt(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn(f"FROM {UV_IMAGE} AS uv", dockerfile)
        self.assertIn("COPY --from=uv /uv /usr/local/bin/uv", dockerfile)
        self.assertNotIn("uv-install.sh", dockerfile)
        self.assertNotIn("apt-get", dockerfile)
        self.assertNotIn("curl", dockerfile)

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

    def test_profile_owns_runtime_paths_while_image_owns_tracing_defaults(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("SHIMPZ_BRAIN_RUNTIME_TOKEN_GID=10016", dockerfile)
        self.assertNotIn("SHIMPZ_BRAIN_RUNTIME_TOKEN_FILE=", dockerfile)
        self.assertNotIn("SHIMPZ_BRAIN_RUNTIME_STATE=", dockerfile)
        self.assertIn("LANGSMITH_TRACING=false", dockerfile)

    def test_runtime_artifact_excludes_the_independent_egress_role(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("agent_runtime.py runtime_api.py model_catalog.json /app/", dockerfile)
        self.assertNotIn("egress/", dockerfile)
        self.assertNotIn("/var/log/brain-egress", dockerfile)
        self.assertNotIn("SHIMPZ_EGRESS_ALLOW", dockerfile)


if __name__ == "__main__":
    unittest.main()
