from __future__ import annotations

import importlib.util
import socket
import sys
import threading
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_proxy(tmp_path: Path):
    root = tmp_path / "http"
    certs = tmp_path / "certs"
    root.mkdir()
    certs.mkdir()
    (certs / "ca.pem").write_bytes(b"sandbox ca\n")
    real_ca = certs / "real-ca.pem"
    real_ca.write_bytes(b"real ca\n")

    path = REPO_ROOT / "scripts" / "sandbox" / "proxy.py"
    name = "sandbox_proxy_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.modules[name] = module
    try:
        sys.argv = [str(path), str(root), str(certs), str(real_ca)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = old_argv
    return module, root


class _RecordingConnection:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def sendall(self, value: bytes) -> None:
        self.sent.append(value)


def test_connect_to_non_fixture_host_uses_raw_tunnel_without_mitm(tmp_path, monkeypatch):
    proxy, _ = _load_proxy(tmp_path)
    connection = _RecordingConnection()
    tunneled: list[tuple[object, str, int]] = []

    monkeypatch.setattr(
        proxy,
        "tunnel_connect",
        lambda conn, host, port: tunneled.append((conn, host, port)),
        raising=False,
    )
    monkeypatch.setattr(
        proxy,
        "cert_for",
        lambda host: pytest.fail(f"non-fixture host was intercepted: {host}"),
    )

    proxy.handle_connect(connection, "registry.npmjs.org:443")

    assert tunneled == [(connection, "registry.npmjs.org", 443)]
    assert connection.sent == []


def test_connect_to_fixture_host_keeps_mitm_interception(tmp_path, monkeypatch):
    proxy, root = _load_proxy(tmp_path)
    (root / "hermes-agent.nousresearch.com").mkdir()
    connection = _RecordingConnection()
    intercepted: list[tuple[object, str, int]] = []

    monkeypatch.setattr(
        proxy,
        "intercept_connect",
        lambda conn, host, port: intercepted.append((conn, host, port)),
        raising=False,
    )
    monkeypatch.setattr(
        proxy,
        "tunnel_connect",
        lambda conn, host, port: pytest.fail(f"fixture host was raw-tunneled: {host}"),
    )

    proxy.handle_connect(connection, "hermes-agent.nousresearch.com:443")

    assert intercepted == [(connection, "hermes-agent.nousresearch.com", 443)]
    assert connection.sent == []


def test_raw_tunnel_drains_upstream_response_after_client_half_close(tmp_path):
    proxy, _ = _load_proxy(tmp_path)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    server_received: list[bytes] = []
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            with listener:
                upstream, _ = listener.accept()
                with upstream:
                    chunks: list[bytes] = []
                    while chunk := upstream.recv(4096):
                        chunks.append(chunk)
                    server_received.append(b"".join(chunks))
                    upstream.sendall(b"upstream response")
        except BaseException as error:  # surfaced in the main test thread
            errors.append(error)

    server_thread = threading.Thread(target=serve)
    server_thread.start()
    client, proxy_side = socket.socketpair()

    def run_proxy() -> None:
        with proxy_side:
            proxy.tunnel_connect(proxy_side, "127.0.0.1", port)

    proxy_thread = threading.Thread(target=run_proxy)
    proxy_thread.start()
    try:
        assert client.recv(4096) == b"HTTP/1.1 200 Connection Established\r\n\r\n"
        client.sendall(b"client request")
        client.shutdown(socket.SHUT_WR)
        assert client.recv(4096) == b"upstream response"
    finally:
        client.close()
        proxy_side.close()
        proxy_thread.join(timeout=2)
        server_thread.join(timeout=2)

    assert errors == []
    assert server_received == [b"client request"]
    assert not proxy_thread.is_alive()
    assert not server_thread.is_alive()


def test_proxy_builds_combined_real_and_fixture_ca_bundle(tmp_path):
    proxy, _ = _load_proxy(tmp_path)

    trust_bundle = proxy.write_trust_bundle()

    assert trust_bundle == proxy.CERTS / "trust-bundle.pem"
    assert trust_bundle.read_bytes() == b"sandbox ca\nreal ca\n"


def test_sandbox_clients_use_combined_real_and_fixture_ca_bundle():
    stage2 = (REPO_ROOT / "scripts" / "sandbox" / "stage2-run.sh").read_text(encoding="utf-8")

    assert "--setenv CURL_CA_BUNDLE /work/certs/trust-bundle.pem" in stage2
    assert "--setenv SSL_CERT_FILE /work/certs/trust-bundle.pem" in stage2
    assert "--setenv GIT_SSL_CAINFO /work/certs/trust-bundle.pem" in stage2
    assert "--setenv NODE_EXTRA_CA_CERTS /work/certs/trust-bundle.pem" in stage2
