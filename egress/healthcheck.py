#!/usr/local/bin/python3
"""Docker HEALTHCHECK probe: the proxy is up AND processing requests correctly.

Prove the control is live, not merely that the port is open. A plain non-CONNECT request must be
refused with 405: this proxy only tunnels CONNECT, never
forward-proxies plain HTTP (which would be an `http://` exfil path). That check is independent of the
catalog-derived host allowlist, so it does not create a provider request or require a live provider.
"""

import socket
import sys

try:
    sock = socket.create_connection(("127.0.0.1", 8888), timeout=3)
    sock.sendall(b"GET / HTTP/1.1\r\nHost: healthcheck\r\n\r\n")
    resp = sock.recv(128)
    sock.close()
except OSError:
    sys.exit(1)
else:
    sys.exit(0 if b" 405 " in resp else 1)  # 405 = CONNECT-only enforcement is live
