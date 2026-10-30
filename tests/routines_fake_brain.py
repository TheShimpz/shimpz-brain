"""A stand-in Rust Brain binary for the Routine eval's process tests, which the run starts as it starts the real one.

``launcher`` writes an executable copy of this file. Started as ``serve --listen HOST:PORT ... --state-dir DIR``, it
records its arguments in ``DIR/argv.json`` and answers ``GET /health`` until it is terminated; ``FAKE_BRAIN_MODE=quiet``
makes it exit at once instead, so it never becomes healthy.
"""

import http.server
import json
import os
import sys
from pathlib import Path


def launcher(directory: Path) -> str:
    """An owner-only executable that runs this file with the current interpreter, wherever the run starts it."""
    path = directory / "shimpz-brain"
    path.write_text(f"#!{sys.executable}\n" + Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    path.chmod(0o700)
    return str(path)


class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        return


def main() -> int:
    arguments = sys.argv[1:]
    if os.environ.get("FAKE_BRAIN_MODE") == "quiet":
        return 0
    state = Path(arguments[arguments.index("--state-dir") + 1])
    (state / "argv.json").write_text(json.dumps(arguments), encoding="utf-8")
    host, port = arguments[arguments.index("--listen") + 1].rsplit(":", 1)
    http.server.HTTPServer((host, int(port)), Health).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
