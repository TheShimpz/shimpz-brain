"""Request-scoped cancellation of one chat turn's provider HTTP I/O (ADR-0079).

A turn runs its synchronous graph inside ``CancelScope.run``. The scope travels in a context variable, which the
LangGraph sync executor copies into its worker threads, so every provider socket read, write, and TLS handshake of
that turn registers with it. ``cancel`` shuts those sockets down to wake a blocked call, and every later connect or
request of the turn fails before it starts. Calls without a scope (routing, planning, labels) are never affected.

``ProviderCallCancelled`` subclasses ``BaseException`` on purpose: the provider SDKs retry every ``Exception`` with
backoff, and the runtime maps every ``Exception`` to a provider failure. The graph still unwinds through its normal
exit, so checkpoint writes finish before the per-thread lock is released. A cancelled call may already be billed.
"""

from __future__ import annotations

import contextvars
import os
import socket
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress

import httpcore
import httpx

# Provider traffic first reaches the local egress proxy, so connecting or waiting for a pooled connection is brief.
CONNECT_TIMEOUT_SECONDS = 5.0
POOL_TIMEOUT_SECONDS = 5.0
MAX_CONNECTIONS = 100
MAX_KEEPALIVE_CONNECTIONS = 20
KEEPALIVE_EXPIRY_SECONDS = 5.0


class ProviderCallCancelled(BaseException):
    """The turn owning this provider call was cancelled."""


def _shutdown(sock: socket.socket) -> None:
    # The base method only issues the shutdown syscall; SSLSocket.shutdown would also detach its TLS state while
    # another thread may still be reading through it.
    with suppress(OSError):
        socket.socket.shutdown(sock, socket.SHUT_RDWR)


class CancelScope:
    """One turn's cancellation state and the provider sockets it is blocked on right now."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._sockets: set[socket.socket] = set()

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def run[T](self, work: Callable[[], T]) -> T:
        token = _SCOPE.set(self)
        try:
            return work()
        finally:
            _SCOPE.reset(token)

    def cancel(self) -> None:
        # Shutting down under the lock keeps every registered socket from leaving the scope, and so from returning
        # to the shared pool, before its shutdown has happened.
        with self._lock:
            self._cancelled = True
            for sock in self._sockets:
                _shutdown(sock)

    def check(self) -> None:
        if self.cancelled:
            raise ProviderCallCancelled

    @contextmanager
    def io(self, sock: socket.socket) -> Iterator[None]:
        with self._lock:
            if self._cancelled:
                raise ProviderCallCancelled
            self._sockets.add(sock)
        try:
            yield
        except Exception as exc:
            if self.cancelled:
                raise ProviderCallCancelled from exc
            raise
        finally:
            with self._lock:
                self._sockets.discard(sock)
        self.check()


_SCOPE: contextvars.ContextVar[CancelScope | None] = contextvars.ContextVar("provider_cancel_scope", default=None)


def _guard(stream: httpcore.NetworkStream) -> AbstractContextManager[None]:
    scope = _SCOPE.get()
    if scope is None:
        return nullcontext()
    return scope.io(stream.get_extra_info("socket"))


class _Stream(httpcore.NetworkStream):
    def __init__(self, inner: httpcore.NetworkStream) -> None:
        self._inner = inner

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        with _guard(self._inner):
            return self._inner.read(max_bytes, timeout)

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        with _guard(self._inner):
            self._inner.write(buffer, timeout)

    def close(self) -> None:
        self._inner.close()

    def start_tls(self, ssl_context, server_hostname: str | None = None, timeout: float | None = None):
        scope = _SCOPE.get()
        if scope is None:
            return _Stream(self._inner.start_tls(ssl_context, server_hostname, timeout))
        # wrap_socket detaches this socket object and handshakes on a new one, so the scope holds a duplicate
        # descriptor of the same connection; shutting it down still wakes the handshake.
        handle = self._inner.get_extra_info("socket").dup()
        try:
            with scope.io(handle):
                return _Stream(self._inner.start_tls(ssl_context, server_hostname, timeout))
        finally:
            handle.close()

    def get_extra_info(self, info: str) -> object:
        return self._inner.get_extra_info(info)


def _bounded(timeout: float | None, bound: float) -> float:
    return bound if timeout is None else min(timeout, bound)


class _Backend(httpcore.SyncBackend):
    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        scope = _SCOPE.get()
        if scope is not None:
            scope.check()
        stream = super().connect_tcp(
            host, port, _bounded(timeout, CONNECT_TIMEOUT_SECONDS), local_address, socket_options
        )
        if scope is not None and scope.cancelled:
            stream.close()
            raise ProviderCallCancelled
        return _Stream(stream)


class _Transport(httpx.HTTPTransport):
    """The default HTTP/1.1 transport over the cancellable backend; pinned to httpx 0.28's pool attribute."""

    def __init__(self, proxy: str | None) -> None:
        super().__init__()
        options = {
            "ssl_context": httpx.create_ssl_context(),
            "max_connections": MAX_CONNECTIONS,
            "max_keepalive_connections": MAX_KEEPALIVE_CONNECTIONS,
            "keepalive_expiry": KEEPALIVE_EXPIRY_SECONDS,
            "network_backend": _Backend(),
        }
        self._pool.close()
        self._pool = (
            httpcore.ConnectionPool(**options)
            if proxy is None
            else httpcore.HTTPProxy(proxy_url=httpcore.URL(proxy), **options)
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        scope = _SCOPE.get()
        if scope is not None:
            scope.check()
        timeout = dict(request.extensions.get("timeout", {}))
        timeout["connect"] = _bounded(timeout.get("connect"), CONNECT_TIMEOUT_SECONDS)
        timeout["pool"] = _bounded(timeout.get("pool"), POOL_TIMEOUT_SECONDS)
        request.extensions["timeout"] = timeout
        return super().handle_request(request)


def client() -> httpx.Client:
    """Build one credential-free provider pool whose calls a turn's scope can cancel."""
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or None
    return httpx.Client(transport=_Transport(proxy), trust_env=False)
