"""A stand-in Brain interpreter for the Routine eval's process tests, which the run starts as it starts Brain's.

``launcher`` writes an executable copy of this file, so ``start_brain`` runs it in place of Brain's interpreter with
Brain's own arguments. It prints port 40000 plus the number in its token file's name and serves until stdin closes;
``FAKE_BRAIN_MODE`` makes it print ``no port``, ``0``, or ``70000`` instead, ``quiet`` print nothing, ``silent`` hang,
or ``dead`` stop Brain 1 right after it says its port.
"""

import os
import sys
import time
from pathlib import Path


def launcher(directory: Path) -> str:
    """An owner-only executable that runs this file with the current interpreter, wherever ``start_brain`` runs it."""
    path = directory / "python"
    path.write_text(f"#!{sys.executable}\n" + Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    path.chmod(0o700)
    return str(path)


def main() -> int:
    token = Path(sys.argv[sys.argv.index("--token-file") + 1])
    number = int(token.stem.rsplit("-", 1)[1])
    mode = os.environ.get("FAKE_BRAIN_MODE", "serve")
    if mode == "silent":
        time.sleep(30)
        return 0
    if mode == "quiet":
        return 0
    print({"no port": "no port", "0": 0, "70000": 70000}.get(mode, 40000 + number), flush=True)
    if mode == "dead" and number == 1:
        return 0
    sys.stdin.read()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
