"""Team's closed start envelope, as its chat contract builds it, and the malformed contents Brain must refuse."""

import json
from collections.abc import Sequence

import memory


def envelope(message: str, files: Sequence[dict[str, object]] = ()) -> str:
    """Team's closed start envelope exactly as Team's chat contract builds it."""
    return json.dumps({"files": list(files), "message": message}, separators=(",", ":"), ensure_ascii=False)


MALFORMED = (
    "not json",
    json.dumps(["files", "message"]),
    json.dumps({"message": "hello"}),
    json.dumps({"files": [], "message": "hello", "extra": 1}),
    json.dumps({"files": {}, "message": "hello"}),
    json.dumps({"files": [], "message": 1}),
    json.dumps({"files": [], "message": "   "}),
    json.dumps({"files": [], "message": "x" * (memory.MAX_TURN_MESSAGE_CHARS + 1)}),
    [{"type": "text", "text": envelope("hello")}],
)
