"""Behavior contracts for admitting an opaque CONNECT tunnel by its TLS ClientHello server name."""

from __future__ import annotations

import json
import socket
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "egress"))

import client_hello

VECTORS = json.loads((TESTS / "egress_client_hello_vectors.json").read_text(encoding="utf-8"))["vectors"]
CURL = bytes.fromhex(VECTORS[0]["flight"])
CHANGE_CIPHER_SPEC = bytes.fromhex("140303000101")


def _length(value: bytes, width: int = 2) -> bytes:
    return len(value).to_bytes(width, "big") + value


def _extension(kind: int, data: bytes) -> bytes:
    return kind.to_bytes(2, "big") + _length(data)


def _sni(*entries: tuple[int, bytes]) -> bytes:
    return _extension(0, _length(b"".join(bytes([kind]) + _length(name) for kind, name in entries)))


def _message(
    *extensions: bytes, session: bytes = b"", suites: bytes = b"\x13\x01", compression: bytes = b"\x00"
) -> bytes:
    """One ClientHello handshake message, header included; no extension block at all when none is given."""
    body = b"\x03\x03" + bytes(32) + _length(session, 1) + _length(suites) + _length(compression, 1)
    if extensions:
        body += _length(b"".join(extensions))
    return b"\x01" + len(body).to_bytes(3, "big") + body


def _records(message: bytes, *sizes: int) -> bytes:
    """Frame a handshake message into handshake records of the given fragment sizes, the rest in one last record."""
    flight, offset = b"", 0
    for size in (*sizes, len(message)):
        fragment = message[offset : offset + size]
        if fragment:
            flight += b"\x16\x03\x01" + _length(fragment)
        offset += len(fragment)
    return flight


def _flight(*extensions: bytes, **fields: bytes) -> bytes:
    return _records(_message(*extensions, **fields))


def _messages(flight: bytes) -> bytes:
    """The handshake bytes of a flight's records, so a real ClientHello can be framed again."""
    out, offset = b"", 0
    while offset < len(flight):
        length = int.from_bytes(flight[offset + 3 : offset + 5], "big")
        out += flight[offset + 5 : offset + 5 + length]
        offset += 5 + length
    return out


class _Chunks:
    """A socket stand-in that yields fixed chunks and records each requested read size."""

    def __init__(self, *chunks: bytes | BaseException) -> None:
        self.chunks = list(chunks)
        self.sizes: list[int] = []
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)

    def recv(self, size: int) -> bytes:
        self.sizes.append(size)
        chunk = self.chunks.pop(0) if self.chunks else b""
        if isinstance(chunk, BaseException):
            raise chunk
        if len(chunk) > size:
            self.chunks.insert(0, chunk[size:])
        return chunk[:size]


def _later() -> float:
    return time.monotonic() + 10


class RealClientTest(unittest.TestCase):
    def test_every_captured_client_is_parsed_and_admitted_only_for_its_own_name(self) -> None:
        for vector in VECTORS:
            flight = bytes.fromhex(vector["flight"])
            with self.subTest(client=vector["client"]):
                if vector["server_name"] is None:
                    with self.assertRaises(client_hello.RefusalError) as refused:
                        client_hello.parse(flight)
                    self.assertEqual(refused.exception.reason, "sni-absent")
                    continue
                self.assertEqual(client_hello.parse(flight), vector["server_name"])
                left, right = socket.socketpair()
                self.addCleanup(left.close)
                self.addCleanup(right.close)
                right.sendall(flight)
                self.assertEqual(client_hello.admit(left, vector["server_name"].upper() + ".", _later()), flight)
                with self.assertRaises(client_hello.RefusalError) as refused:
                    client_hello.admit(_Chunks(flight), "fronted.example", _later())
                self.assertEqual(refused.exception.reason, "sni-mismatch")

    def test_every_strict_prefix_of_a_real_flight_waits_for_more(self) -> None:
        for vector in VECTORS:
            flight = bytes.fromhex(vector["flight"])
            with self.subTest(client=vector["client"]):
                for end in range(len(flight)):
                    self.assertIsNone(client_hello.parse(flight[:end]))

    def test_a_client_hello_split_across_records_and_reads_is_reassembled(self) -> None:
        message = _messages(CURL)
        for sizes in ((1,), (2, 1, 1), (3, 500), (1,) * 40, (len(message) - 1,)):
            with self.subTest(sizes=sizes[:3]):
                self.assertEqual(client_hello.parse(_records(message, *sizes)), "api.cloudflare.com")
        reads = _Chunks(*(CURL[index : index + 1] for index in range(len(CURL))))
        self.assertEqual(client_hello.admit(reads, "api.cloudflare.com", _later()), CURL)
        self.assertEqual(len(reads.sizes), len(CURL))

    def test_records_after_the_client_hello_are_ignored_by_the_decision_and_forwarded(self) -> None:
        trailing = CURL + CHANGE_CIPHER_SPEC + b"\x17\x03\x03\x00\x02ab"
        self.assertEqual(client_hello.parse(trailing), "api.cloudflare.com")
        self.assertEqual(client_hello.parse(CURL + b"GET / HTTP/1.1"), "api.cloudflare.com")
        self.assertEqual(client_hello.admit(_Chunks(trailing), "api.cloudflare.com", _later()), trailing)

    def test_mutated_flights_only_ever_wait_refuse_or_name_a_server(self) -> None:
        for vector in VECTORS:
            flight = bytes.fromhex(vector["flight"])
            for position in range(0, len(flight), 7):
                for value in (0x00, 0xFF, flight[position] ^ 0x80):
                    mutated = flight[:position] + bytes([value]) + flight[position + 1 :]
                    for end in (position + 1, len(mutated)):
                        try:
                            result = client_hello.parse(mutated[:end])
                        except client_hello.RefusalError as refused:
                            self.assertTrue(refused.reason)
                        else:
                            self.assertTrue(result is None or isinstance(result, str))


class RefusalTest(unittest.TestCase):
    def refused(self, flight: bytes, reason: str) -> None:
        with self.assertRaises(client_hello.RefusalError) as refused:
            client_hello.parse(flight)
        self.assertEqual(refused.exception.reason, reason)

    def test_a_first_flight_that_is_not_a_tls_handshake_is_refused_from_its_first_bytes(self) -> None:
        for flight in (
            b"G",
            b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n",
            b"SSH-2.0-OpenSSH_9.9\r\n",
            b"\x80\x2e\x01\x00\x02",  # an SSLv2-format ClientHello
            b"\x17\x03\x03\x00\x05hello",  # application data before any handshake
            b"\x16\x02",  # not a TLS record version
            _records(b"\x02\x00\x00\x04" + bytes(4)),  # a handshake message other than ClientHello
        ):
            with self.subTest(flight=flight[:12]):
                self.refused(flight, "tls-required")

    def test_malformed_record_framing_is_refused(self) -> None:
        message = _message(_sni((0, b"example.com")))
        cases = (
            b"\x16\x03\x01\x00\x00" + _records(message),  # an empty handshake fragment
            b"\x16\x03\x01\x40\x01" + bytes(16385),  # a record beyond 2^14
            _records(message, 10)[:15] + b"\x15\x03\x03\x00\x02\x02\x28",  # another record type inside the message
            b"\x16\x03\x01" + _length(message + b"\x01\x00\x00\x00"),  # handshake bytes after the ClientHello
        )
        for flight in cases:
            with self.subTest(flight=flight[:8]):
                self.refused(flight, "tls-malformed")

    def test_malformed_client_hello_bodies_are_refused(self) -> None:
        sni = _sni((0, b"example.com"))
        cases = (
            _records(b"\x01\x00\x00\x10" + bytes(16)),  # shorter than version and random
            _flight(sni, session=bytes(33)),
            _flight(sni, suites=b""),
            _flight(sni, suites=b"\x13"),
            _flight(sni, compression=b""),
            _flight(b"\x00\x0a\x00\x10\x00"),  # an extension overruns its block
            _records(
                b"\x01" + (len(_message(sni)) - 3).to_bytes(3, "big") + _message(sni)[4:] + b"\x00"
            ),  # a byte after the extension block
            _flight(sni, sni),  # duplicate server_name extensions
            _flight(_extension(10, b"\x00\x02\x00\x1d"), _extension(10, b"\x00\x02\x00\x1d"), sni),
            _flight(_sni((0, b"example.com"), (0, b"other.example"))),
            _flight(_sni((0, b"example.com"), (1, b"other"))),
            _flight(_sni((1, b"example.com"))),
            _flight(_extension(0, _length(b"\x00" + _length(b"example.com")) + b"\x00")),
            _flight(_sni((0, b""))),
            _flight(_sni((0, b"example.com."))),
            _flight(_sni((0, b"203.0.113.7"))),
            _flight(_sni((0, b"2001:db8::7"))),
            _flight(_sni((0, "exämple.com".encode()))),
        )
        for index, flight in enumerate(cases):
            with self.subTest(case=index):
                self.refused(flight, "tls-malformed")

    def test_a_missing_server_name_is_refused(self) -> None:
        self.refused(_flight(), "sni-absent")
        self.refused(_flight(_extension(10, b"\x00\x02\x00\x1d")), "sni-absent")
        self.refused(_flight(_extension(0, b"\x00\x00")), "sni-absent")
        self.refused(_flight(_extension(0, b"")), "tls-malformed")

    def test_encrypted_client_hello_is_refused_even_beside_a_matching_name(self) -> None:
        for kind in (0xFE0D, 0xFFCE):
            for extensions in (
                (_sni((0, b"example.com")), _extension(kind, bytes(40))),
                (_extension(kind, b""), _sni((0, b"example.com"))),
            ):
                with self.subTest(kind=hex(kind)):
                    self.refused(_flight(*extensions), "ech-refused")

    def test_the_flight_is_bounded_whatever_its_framing(self) -> None:
        def padded(total: int) -> bytes:
            """A one-record flight of exactly `total` bytes."""
            base = _message(_sni((0, b"example.com")), _extension(21, b""))
            pad = total - client_hello.RECORD_HEADER - len(base)
            return _records(_message(_sni((0, b"example.com")), _extension(21, bytes(pad))))

        at_cap = padded(client_hello.MAX_FLIGHT_BYTES)
        self.assertEqual(len(at_cap), client_hello.MAX_FLIGHT_BYTES)
        self.assertEqual(client_hello.admit(_Chunks(at_cap), "example.com", _later()), at_cap)
        over_cap = _records(_messages(padded(client_hello.MAX_FLIGHT_BYTES)), 1000)
        reads = _Chunks(over_cap)
        with self.assertRaises(client_hello.RefusalError) as refused:
            client_hello.admit(reads, "example.com", _later())
        self.assertEqual(refused.exception.reason, "tls-oversize")
        self.assertEqual(reads.sizes, [client_hello.MAX_FLIGHT_BYTES])
        self.refused(b"\x16\x03\x01\x00\x04\x01\x00\x40\x00", "tls-oversize")


class ReadTest(unittest.TestCase):
    def incomplete(self, sock: object, deadline: float | None = None) -> None:
        with self.assertRaises(client_hello.RefusalError) as refused:
            client_hello.admit(sock, "api.cloudflare.com", _later() if deadline is None else deadline)
        self.assertEqual(refused.exception.reason, "tls-incomplete")

    def test_a_closed_failed_or_late_client_is_incomplete(self) -> None:
        self.incomplete(_Chunks(CURL[:100], b""))
        self.incomplete(_Chunks(CURL[:100], OSError("reset")))
        late = _Chunks(CURL)
        self.incomplete(late, time.monotonic() - 1)
        self.assertEqual(late.sizes, [])

    def test_a_trickle_is_cut_at_one_absolute_deadline(self) -> None:
        clock = iter([100.0, 104.0, 111.0])
        trickle = _Chunks(CURL[:1], CURL[1:2], CURL[2:])
        with mock.patch.object(client_hello.time, "monotonic", side_effect=lambda: next(clock)):
            self.incomplete(trickle, 110.0)
        self.assertEqual(trickle.timeouts, [10.0, 6.0])
        self.assertEqual(trickle.sizes, [client_hello.MAX_FLIGHT_BYTES, client_hello.MAX_FLIGHT_BYTES - 1])

    def test_a_refused_first_byte_stops_reading(self) -> None:
        reads = _Chunks(b"GET / HTTP/1.1\r\n", CURL)
        with self.assertRaises(client_hello.RefusalError) as refused:
            client_hello.admit(reads, "api.cloudflare.com", _later())
        self.assertEqual(refused.exception.reason, "tls-required")
        self.assertEqual(len(reads.sizes), 1)


if __name__ == "__main__":
    unittest.main()
