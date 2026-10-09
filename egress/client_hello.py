"""Admit an opaque CONNECT tunnel only when the client's TLS ClientHello names exactly the host the proxy allowed.

A CONNECT proxy that never terminates TLS decides on the CONNECT target alone. Without this check a client allowed
one host on a shared CDN edge could put another name in its TLS server_name (SNI) and reach a different site behind
the same address: domain fronting. The proxy therefore reads the client's first handshake flight and requires
exactly one host_name SNI, in the RFC 6066 section 3 form (ASCII, no trailing dot, no IP literal), equal to the
CONNECT host case-insensitively, and no encrypted_client_hello (RFC 9849) or its ESNI predecessor, whose outer name
is not the name the destination routes on. Only then are those bytes forwarded unchanged. Nothing is decrypted,
terminated, or altered, and no certificate is held.

The guarantee is first-ClientHello SNI admission. RFC 8446 section 4.1.2 forbids a client from changing server_name
in a ClientHello sent after a HelloRetryRequest, and RFC 6066 forbids resuming a session under another name; a
client that breaks those rules, renegotiates TLS 1.2, or names another site inside the encrypted stream (an HTTP Host
header or HTTP/2 connection reuse) is refused only by a destination that enforces them.

This module is pure stdlib and owns no policy, audit, or identity. Each proxy ships its own copy, decides when to
call it, and records the refusal in its own audit.
"""

import ipaddress
import socket
import time

MAX_FLIGHT_BYTES = 16 * 1024  # read before deciding; a real ClientHello is 0.25 to 1.6 KiB
RECORD_HEADER = 5
MAX_RECORD = 1 << 14  # RFC 8446 section 5.1
HANDSHAKE = 22
TLS_MAJOR = 3
CLIENT_HELLO = 1
SERVER_NAME = 0
HOST_NAME = 0
# encrypted_client_hello (RFC 9849 section 11.1) and the encrypted_server_name of its ESNI drafts.
REFUSED_EXTENSIONS = frozenset({0xFE0D, 0xFFCE})


class RefusalError(Exception):
    """The tunnel's first flight is not an admissible ClientHello for the allowed host; `reason` is audit-stable."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def admit(sock: socket.socket, host: str, deadline: float) -> bytes:
    """Read the client's first flight until its ClientHello ends and return every byte read, unchanged.

    `deadline` is absolute and is never extended per read. Raises RefusalError when the flight is not TLS, malformed,
    larger than MAX_FLIGHT_BYTES, incomplete by the deadline, or names any server other than `host`. The caller
    forwards the returned bytes before splicing the rest of the tunnel.
    """
    flight = bytearray()
    while (server_name := parse(flight)) is None:
        if len(flight) >= MAX_FLIGHT_BYTES:
            raise RefusalError("tls-oversize")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RefusalError("tls-incomplete")
        try:
            sock.settimeout(remaining)
            chunk = sock.recv(MAX_FLIGHT_BYTES - len(flight))
        except OSError:
            raise RefusalError("tls-incomplete") from None
        if not chunk:
            raise RefusalError("tls-incomplete")
        flight.extend(chunk)
    if server_name != host.lower().removesuffix("."):
        raise RefusalError("sni-mismatch")
    return bytes(flight)


def parse(flight: bytes | bytearray) -> str | None:
    """Return the lowercased SNI host name of the ClientHello that starts `flight`, or None while it is incomplete.

    The decision rests only on the bytes up to the record that ends the ClientHello; anything after it, such as
    0-RTT or change_cipher_spec records, is ignored here and forwarded by the caller. Raises RefusalError on anything a
    conforming client's first flight never contains.
    """
    message = _client_hello(flight)
    return None if message is None else _server_name(message)


def _client_hello(flight: bytes | bytearray) -> bytes | None:
    """Reassemble the first handshake message across records; None until a record boundary completes it."""
    handshake = bytearray()
    offset = 0
    while (fragment := _fragment(flight, offset)) is not None:
        handshake.extend(fragment)
        offset += RECORD_HEADER + len(fragment)
        if handshake[0] != CLIENT_HELLO:
            raise RefusalError("tls-required")
        if len(handshake) < 4:
            continue
        size = 4 + _number(handshake, 1, 3)
        if size > MAX_FLIGHT_BYTES:
            raise RefusalError("tls-oversize")
        if len(handshake) == size:
            return bytes(handshake[4:])
        if len(handshake) > size:  # RFC 8446 section 5.1: a ClientHello ends on a record boundary
            raise RefusalError("tls-malformed")
    return None


def _fragment(flight: bytes | bytearray, offset: int) -> bytes | None:
    """Return the fragment of the handshake record at `offset`, or None while that record is incomplete."""
    header = flight[offset : offset + RECORD_HEADER]
    if (header and header[0] != HANDSHAKE) or (len(header) > 1 and header[1] != TLS_MAJOR):
        raise RefusalError("tls-malformed" if offset else "tls-required")
    if len(header) < RECORD_HEADER:
        return None
    length = _number(header, 3, 2)
    if not 1 <= length <= MAX_RECORD:  # RFC 8446 section 5.1: no empty handshake fragment, no overflow
        raise RefusalError("tls-malformed")
    fragment = flight[offset + RECORD_HEADER : offset + RECORD_HEADER + length]
    return bytes(fragment) if len(fragment) == length else None


def _server_name(body: bytes) -> str:
    """Validate the ClientHello body layout (RFC 8446 section 4.1.2) and return its one SNI host name."""
    reader = _Reader(body)
    reader.take(2 + 32)  # legacy_version, random
    if len(reader.vector(1)) > 32:  # legacy_session_id
        raise RefusalError("tls-malformed")
    suites = reader.vector(2)
    if not suites or len(suites) % 2:
        raise RefusalError("tls-malformed")
    if not reader.vector(1):  # legacy_compression_methods
        raise RefusalError("tls-malformed")
    if reader.done():  # a ClientHello without extensions carries no server_name
        raise RefusalError("sni-absent")
    extensions = _Reader(reader.vector(2))
    reader.end()
    seen: set[int] = set()
    names: bytes | None = None
    while not extensions.done():
        kind = _number(extensions.take(2), 0, 2)
        data = extensions.vector(2)
        if kind in seen:  # RFC 8446 section 4.2: at most one extension of each type
            raise RefusalError("tls-malformed")
        seen.add(kind)
        if kind in REFUSED_EXTENSIONS:
            raise RefusalError("ech-refused")
        if kind == SERVER_NAME:
            names = data
    if names is None:
        raise RefusalError("sni-absent")
    return _host_name(names)


def _host_name(extension: bytes) -> str:
    """Return the only entry of a server_name extension, which must be one RFC 6066 host_name."""
    outer = _Reader(extension)
    entries = _Reader(outer.vector(2))
    outer.end()
    if entries.done():
        raise RefusalError("sni-absent")
    if entries.take(1)[0] != HOST_NAME:
        raise RefusalError("tls-malformed")
    raw = entries.vector(2)
    if not entries.done():  # exactly one name: a second entry of any type is refused
        raise RefusalError("tls-malformed")
    try:
        name = raw.decode("ascii")
    except UnicodeDecodeError:
        raise RefusalError("tls-malformed") from None
    # RFC 6066 section 3: no trailing dot, and literal IPv4 and IPv6 addresses are not permitted.
    if not name or name.endswith(".") or _is_address(name):
        raise RefusalError("tls-malformed")
    return name.lower()


def _is_address(name: str) -> bool:
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def _number(data: bytes | bytearray, offset: int, width: int) -> int:
    return int.from_bytes(data[offset : offset + width], "big")


class _Reader:
    """Bounds-checked reads over one complete structure; any overrun or leftover byte is malformed."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def take(self, count: int) -> bytes:
        end = self._offset + count
        if end > len(self._data):
            raise RefusalError("tls-malformed")
        chunk = self._data[self._offset : end]
        self._offset = end
        return chunk

    def vector(self, width: int) -> bytes:
        return self.take(_number(self.take(width), 0, width))

    def done(self) -> bool:
        return self._offset == len(self._data)

    def end(self) -> None:
        if not self.done():
            raise RefusalError("tls-malformed")
