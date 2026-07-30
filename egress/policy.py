"""Fail-closed Brain provider policy derived only from the packaged model catalog."""

from __future__ import annotations

import json
import re
import stat
from pathlib import Path

CATALOG_PATH = Path("/app/model_catalog.json")
MAX_CATALOG_BYTES = 256 * 1024
ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
HOST_RE = re.compile(
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z"
)


class ProviderPolicyError(RuntimeError):
    """The packaged provider catalog cannot safely authorize Brain egress."""


def _exact_object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ProviderPolicyError(f"{label} is invalid")
    return value


def _bounded_text(value: object, maximum: int, label: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise ProviderPolicyError(f"{label} is invalid")
    return value


def _provider_endpoint(value: object) -> str:
    endpoint = _exact_object(value, {"host", "path"}, "provider endpoint")
    host = _bounded_text(endpoint["host"], 253, "provider host")
    path = _bounded_text(endpoint["path"], 256, "provider path")
    if HOST_RE.fullmatch(host) is None or "*" in host or not path.startswith("/"):
        raise ProviderPolicyError("provider endpoint is invalid")
    return host


def _model_ids(value: object) -> frozenset[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= 128:
        raise ProviderPolicyError("provider models are invalid")
    model_ids: set[str] = set()
    for raw_model in value:
        model = _exact_object(
            raw_model,
            {
                "id",
                "input_usd_per_million_cents",
                "output_usd_per_million_cents",
                "title",
            },
            "provider model",
        )
        model_id = _bounded_text(model["id"], 128, "model id")
        if ID_RE.fullmatch(model_id) is None or model_id in model_ids:
            raise ProviderPolicyError("model id is invalid")
        _bounded_text(model["title"], 120, "model title")
        for field in ("input_usd_per_million_cents", "output_usd_per_million_cents"):
            price = model[field]
            if not isinstance(price, int) or isinstance(price, bool) or not 0 <= price <= 10_000_000:
                raise ProviderPolicyError("model price is invalid")
        model_ids.add(model_id)
    return frozenset(model_ids)


def _provider(value: object, provider_ids: set[str]) -> tuple[str, str]:
    provider = _exact_object(
        value,
        {"credential_validation", "default_model", "id", "models", "title"},
        "provider",
    )
    provider_id = _bounded_text(provider["id"], 128, "provider id")
    if ID_RE.fullmatch(provider_id) is None or provider_id in provider_ids:
        raise ProviderPolicyError("provider id is invalid")
    _bounded_text(provider["title"], 80, "provider title")
    model_ids = _model_ids(provider["models"])
    if provider["default_model"] not in model_ids:
        raise ProviderPolicyError("provider default model is invalid")
    return provider_id, _provider_endpoint(provider["credential_validation"])


def _catalog_bytes(path: Path) -> bytes:
    try:
        metadata = path.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or not 1 <= metadata.st_size <= MAX_CATALOG_BYTES
        ):
            raise ProviderPolicyError("provider catalog metadata is invalid")
        return path.read_bytes()
    except OSError as exc:
        raise ProviderPolicyError("provider catalog is unavailable") from exc


def load_provider_hosts(path: Path = CATALOG_PATH) -> frozenset[str]:
    """Return the exact provider hosts from one valid immutable catalog."""
    try:
        raw_catalog = json.loads(_catalog_bytes(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderPolicyError("provider catalog is invalid") from exc
    catalog = _exact_object(raw_catalog, {"default_provider", "providers", "schema"}, "provider catalog")
    if catalog["schema"] != 1 or isinstance(catalog["schema"], bool):
        raise ProviderPolicyError("provider catalog schema is invalid")
    providers = catalog["providers"]
    if not isinstance(providers, list) or not 1 <= len(providers) <= 16:
        raise ProviderPolicyError("provider catalog providers are invalid")
    provider_ids: set[str] = set()
    hosts: set[str] = set()
    for value in providers:
        provider_id, host = _provider(value, provider_ids)
        provider_ids.add(provider_id)
        hosts.add(host)
    if catalog["default_provider"] not in provider_ids or not hosts:
        raise ProviderPolicyError("provider catalog default is invalid")
    return frozenset(hosts)
