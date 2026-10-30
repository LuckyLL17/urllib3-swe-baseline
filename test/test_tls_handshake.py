from __future__ import annotations

import contextlib
import socket
import ssl
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from .conftest import ServerConfig

import urllib3
from dummyserver.socketserver import DEFAULT_CA, DEFAULT_CERTS
from urllib3.connection import HTTPSConnection
from urllib3.exceptions import MaxRetryError, ProxyError, SSLError
from urllib3.tls import (
    TLSHandshakeRecorder,
    TLSHandshakeResult,
    TLSHandshakeStage,
    classify_failure_stage,
    restore_ssl_context,
    snapshot_ssl_context,
)
from urllib3.util import ssl_

RESULT_FIELDS = TLSHandshakeResult._fields


def _require(result: TLSHandshakeResult | None) -> TLSHandshakeResult:
    assert result is not None
    return result


def _bind_server() -> tuple[socket.socket, int]:
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    return srv, srv.getsockname()[1]


def _quiet_close(*socks: socket.socket | None) -> None:
    for sock in socks:
        if sock is not None:
            with contextlib.suppress(Exception):
                sock.close()


@contextlib.contextmanager
def tls_origin(
    *, alpn: list[str] | None = None, close_after_response: bool = False
) -> Iterator[int]:
    """Raw TLS origin speaking HTTP/1.1, keep-alive by default."""
    srv, port = _bind_server()
    clients: set[socket.socket] = set()
    clients_lock = threading.Lock()

    def serve_conn(cli: socket.socket) -> None:
        tls_sock: ssl.SSLSocket | None = None
        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(DEFAULT_CERTS["certfile"], DEFAULT_CERTS["keyfile"])
            if alpn is not None:
                ctx.set_alpn_protocols(alpn)
            tls_sock = ctx.wrap_socket(cli, server_side=True)
            while True:
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = tls_sock.recv(65536)
                    if not chunk:
                        return
                    data += chunk
                headers = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                if close_after_response:
                    headers += b"Connection: close\r\n"
                tls_sock.sendall(headers + b"\r\nhi")
                if close_after_response:
                    return
        except Exception:
            pass
        finally:
            _quiet_close(tls_sock, cli)

    def run() -> None:
        while True:
            try:
                cli, _ = srv.accept()
            except OSError:
                return
            with clients_lock:
                clients.add(cli)
            threading.Thread(target=serve_conn, args=(cli,), daemon=True).start()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        srv.close()
        with clients_lock:
            _quiet_close(*clients)


@contextlib.contextmanager
def closing_tcp_server() -> Iterator[int]:
    """Accepts one TCP connection and immediately closes it."""
    srv, port = _bind_server()

    def run() -> None:
        cli = None
        try:
            cli, _ = srv.accept()
        except Exception:
            pass
        finally:
            _quiet_close(cli, srv)

    threading.Thread(target=run, daemon=True).start()
    yield port


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except Exception:
        pass


@contextlib.contextmanager
def connect_proxy(proxy_scheme: str, *, status: int = 200) -> Iterator[int]:
    """Minimal HTTP/HTTPS CONNECT proxy. Tunnels to the CONNECT target."""
    srv, port = _bind_server()

    def handle(cli: socket.socket) -> None:
        sock: socket.socket = cli
        upstream: socket.socket | None = None
        try:
            if proxy_scheme == "https":
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(DEFAULT_CERTS["certfile"], DEFAULT_CERTS["keyfile"])
                sock = ctx.wrap_socket(cli, server_side=True)
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = sock.recv(65536)
                if not chunk:
                    return
                data += chunk
            request_line = data.split(b"\r\n", 1)[0].decode()
            _, authority, _ = request_line.split(" ", 2)
            target_host, target_port = authority.rsplit(":", 1)
            if status != 200:
                sock.sendall(
                    f"HTTP/1.1 {status} Forbidden\r\nContent-Length: 0\r\n\r\n".encode()
                )
                return
            upstream = socket.create_connection((target_host, int(target_port)))
            sock.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            t1 = threading.Thread(target=_pipe, args=(sock, upstream), daemon=True)
            t2 = threading.Thread(target=_pipe, args=(upstream, sock), daemon=True)
            t1.start()
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)
        except Exception:
            pass
        finally:
            _quiet_close(upstream, sock, cli)

    def run() -> None:
        while True:
            try:
                cli, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=handle, args=(cli,), daemon=True).start()

    threading.Thread(target=run, daemon=True).start()
    try:
        yield port
    finally:
        srv.close()


def _hostport(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _assert_success(result: TLSHandshakeResult, *, host: str) -> None:
    assert isinstance(result, TLSHandshakeResult)
    assert result.target_host == host
    assert result.tls_completed is True
    assert result.cert_verified is True
    assert result.failure_stage is TLSHandshakeStage.NONE
    assert result.tls_version in ("TLSv1.2", "TLSv1.3")
    assert result.cipher is not None and len(result.cipher) == 3
    assert "http/1.1" in result.alpn_offered
    assert result.alpn_negotiated == "http/1.1"
    assert result.alpn_mismatch is False
    assert result.closed is False
    assert result.peer_cert_fingerprint is not None
    assert len(result.peer_cert_fingerprint) == 64
    int(result.peer_cert_fingerprint, 16)


class TestTLSHandshakeStructure:
    def test_stage_values_are_stable_strings(self) -> None:
        assert TLSHandshakeStage.NONE.value == "none"
        assert TLSHandshakeStage.TCP_CONNECT.value == "tcp_connect"
        assert TLSHandshakeStage.PROXY_TLS.value == "proxy_tls"
        assert TLSHandshakeStage.PROXY_CONNECT.value == "proxy_connect"
        assert TLSHandshakeStage.TLS.value == "tls"
        assert TLSHandshakeStage.CERTIFICATE.value == "certificate"
        assert TLSHandshakeStage.HOSTNAME.value == "hostname"
        assert TLSHandshakeStage.ALPN.value == "alpn"
        assert TLSHandshakeStage.CLOSED.value == "closed"

    def test_recorder_monotonic_identity(self) -> None:
        recorder = TLSHandshakeRecorder()
        assert recorder.latest() is None
        id1 = recorder.new_connection_id()
        id2 = recorder.new_connection_id()
        assert id2 > id1

    def test_result_is_immutable(self) -> None:
        with tls_origin() as port:
            conn = HTTPSConnection(
                "localhost", port, ca_certs=DEFAULT_CA, tls_handshake_reporting=True
            )
            conn.connect()
            result = conn.tls_handshake_result
            conn.close()
        assert result is not None
        with pytest.raises(AttributeError):
            result.tls_completed = False  # type: ignore[misc]

    @pytest.mark.parametrize(
        "reason,expected",
        [
            ("CERTIFICATE_VERIFY_FAILED", TLSHandshakeStage.CERTIFICATE),
            ("self signed certificate", TLSHandshakeStage.CERTIFICATE),
            ("NO_APPLICATION_PROTOCOL", TLSHandshakeStage.ALPN),
            ("UNEXPECTED_EOF_WHILE_READING", TLSHandshakeStage.CLOSED),
            ("EOF occurred in violation", TLSHandshakeStage.CLOSED),
            ("CONNECTION_RESET", TLSHandshakeStage.CLOSED),
            ("HANDSHAKE_FAILURE", TLSHandshakeStage.TLS),
        ],
    )
    def test_classify_failure_stage(self, reason: str, expected: str) -> None:
        err = ssl.SSLError(1, reason)
        err.reason = reason
        assert classify_failure_stage(err) is expected

    def test_snapshot_and_restore_user_context(self) -> None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        snapshot = snapshot_ssl_context(context)
        assert snapshot is not None
        # urllib3-style mutation during connect
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        restore_ssl_context(context, snapshot)
        assert context.verify_mode == ssl.CERT_NONE
        assert context.check_hostname is False


class TestTLSHandshakeConnection:
    def test_direct_success_full_report(self) -> None:
        with tls_origin(alpn=["http/1.1"]) as port:
            conn = HTTPSConnection(
                "localhost", port, ca_certs=DEFAULT_CA, tls_handshake_reporting=True
            )
            conn.connect()
            result = _require(conn.tls_handshake_result)
            conn.close()
            _assert_success(result, host="localhost")
            assert result.sni == "localhost"
            assert result.used_proxy is False
            assert result.proxy_scheme is None
            assert result.proxy_tunnel_established is False
            assert result.proxy_result is None
            assert result.connection_id > 0

    def test_no_private_material_in_result(self) -> None:
        with tls_origin(alpn=["http/1.1"]) as port:
            conn = HTTPSConnection(
                "localhost", port, ca_certs=DEFAULT_CA, tls_handshake_reporting=True
            )
            conn.connect()
            result = conn.tls_handshake_result
            rendered = repr(result)
            conn.close()
        assert "PRIVATE KEY" not in rendered
        assert "BEGIN CERTIFICATE" not in rendered

    def test_certificate_verification_failure(self) -> None:
        with tls_origin() as port:
            conn = HTTPSConnection(
                "localhost", port, tls_handshake_reporting=True
            )  # trusts nothing
            # Direct HTTPSConnection.connect() surfaces the raw stdlib error.
            with pytest.raises(ssl.SSLError):
                conn.connect()
            result = conn.tls_handshake_result
            conn.close()
            assert result is not None
            assert result.tls_completed is False
            assert result.cert_verified is False
            assert result.failure_stage is TLSHandshakeStage.CERTIFICATE
            assert result.tls_version is None
            assert result.cipher is None
            assert result.alpn_negotiated is None
            assert result.peer_cert_fingerprint is None
            assert result.failure_stage is TLSHandshakeStage.CERTIFICATE

    def test_hostname_mismatch(self) -> None:
        from urllib3.util.ssl_match_hostname import CertificateError

        with tls_origin() as port:
            conn = HTTPSConnection(
                "127.0.0.1",
                port,
                ca_certs=DEFAULT_CA,
                assert_hostname="example.com",
                tls_handshake_reporting=True,
            )
            with pytest.raises(CertificateError):
                conn.connect()
            result = conn.tls_handshake_result
            conn.close()
            assert result is not None
            assert result.failure_stage is TLSHandshakeStage.HOSTNAME
            assert result.cert_verified is False
            # TLS itself completed before urllib3's hostname check failed.
            assert result.tls_completed is True
            assert result.tls_version is not None
            assert result.peer_cert_fingerprint is not None

    def test_fingerprint_mismatch(self) -> None:
        with tls_origin() as port:
            conn = HTTPSConnection(
                "localhost",
                port,
                cert_reqs="CERT_NONE",
                assert_fingerprint="00" * 32,
                tls_handshake_reporting=True,
            )
            # assert_fingerprint raises urllib3's own SSLError.
            with pytest.raises(SSLError):
                conn.connect()
            result = conn.tls_handshake_result
            conn.close()
            assert result is not None
            assert result.failure_stage is TLSHandshakeStage.CERTIFICATE
            assert result.tls_completed is True
            assert result.peer_cert_fingerprint is not None
            assert result.peer_cert_fingerprint != "0" * 64

    def test_peer_closed_during_handshake(self) -> None:
        with closing_tcp_server() as port:
            conn = HTTPSConnection("localhost", port, tls_handshake_reporting=True)
            with pytest.raises(ssl.SSLError):
                conn.connect()
            result = conn.tls_handshake_result
            conn.close()
            assert result is not None
            assert result.tls_completed is False
            assert result.failure_stage is TLSHandshakeStage.CLOSED
            assert result.tls_version is None

    def test_tcp_connect_failure(self) -> None:
        # Port with nothing listening on it.
        srv, port = _bind_server()
        srv.close()
        conn = HTTPSConnection("localhost", port, tls_handshake_reporting=True)
        with pytest.raises(Exception):
            conn.connect()
        result = conn.tls_handshake_result
        conn.close()
        assert result is not None
        assert result.tls_completed is False
        assert result.failure_stage is TLSHandshakeStage.TCP_CONNECT

    def test_close_and_repeated_close(self) -> None:
        with tls_origin(alpn=["http/1.1"]) as port:
            conn = HTTPSConnection(
                "localhost", port, ca_certs=DEFAULT_CA, tls_handshake_reporting=True
            )
            conn.connect()
            connection_id = _require(conn.tls_handshake_result).connection_id
            conn.close()
            first = _require(conn.tls_handshake_result)
            assert first.closed is True
            assert first.connection_id == connection_id
            conn.close()  # must not raise
            second = _require(conn.tls_handshake_result)
            assert second.closed is True
            assert second.connection_id == connection_id

    def test_user_context_not_modified_on_success(self) -> None:
        with tls_origin(alpn=["http/1.1"]) as port:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            before = (context.verify_mode, context.check_hostname)
            conn = HTTPSConnection(
                "localhost",
                port,
                ssl_context=context,
                tls_handshake_reporting=True,
            )
            conn.connect()
            after = (context.verify_mode, context.check_hostname)
            conn.close()
        assert before == after

    def test_user_context_not_modified_on_failure(self) -> None:
        with tls_origin() as port:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            before = (context.verify_mode, context.check_hostname)
            conn = HTTPSConnection(
                "localhost",
                port,
                ssl_context=context,  # system CAs -> verification fails
                tls_handshake_reporting=True,
            )
            with pytest.raises(ssl.SSLError):
                conn.connect()
            after = (context.verify_mode, context.check_hostname)
            conn.close()
        assert before == after

    def test_reporting_failure_does_not_break_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from urllib3 import tls as tls_module

        monkeypatch.setattr(
            tls_module, "_read_negotiated_info", lambda sock: None, raising=True
        )
        with tls_origin(alpn=["http/1.1"]) as port:
            with urllib3.HTTPSConnectionPool(
                "localhost",
                port,
                ca_certs=DEFAULT_CA,
                retries=False,
                tls_handshake_reporting=True,
            ) as pool:
                response = pool.request("GET", "/")
                assert response.status == 200
                result = pool.get_tls_handshake_result()
                assert result is not None
                assert result.tls_completed is True
                assert result.tls_version is None

    def test_broken_recorder_does_not_change_outcome(self) -> None:
        with tls_origin(alpn=["http/1.1"]) as port:
            recorder = TLSHandshakeRecorder()

            def explode(result: TLSHandshakeResult) -> TLSHandshakeResult:
                raise RuntimeError("recorder down")

            recorder.record = explode  # type: ignore[method-assign]
            with urllib3.HTTPSConnectionPool(
                "localhost",
                port,
                ca_certs=DEFAULT_CA,
                retries=False,
                tls_handshake_reporting=True,
                _tls_recorder=recorder,
            ) as pool:
                response = pool.request("GET", "/")
                assert response.status == 200


class TestTLSHandshakePoolAndManager:
    def test_connection_pool_manager_response_consistent(self) -> None:
        with tls_origin(alpn=["http/1.1"]) as port:
            manager = urllib3.PoolManager(
                ca_certs=DEFAULT_CA,
                retries=False,
                tls_handshake_reporting=True,
            )
            url = f"https://localhost:{port}/"
            response = manager.request("GET", url)
            result = _require(response.tls_handshake_result)
            pool = manager.connection_from_host("localhost", port, "https")
            assert result == pool.get_tls_handshake_result()
            assert result == manager.get_tls_handshake_result()
            assert result == manager.get_tls_handshake_result(url)

    def test_response_snapshot_survives_connection_reuse(self) -> None:
        with tls_origin(alpn=["http/1.1"]) as port:
            manager = urllib3.PoolManager(
                ca_certs=DEFAULT_CA,
                retries=False,
                tls_handshake_reporting=True,
            )
            url = f"https://localhost:{port}/"
            first = manager.request("GET", url)
            first_id = _require(first.tls_handshake_result).connection_id
            second = manager.request("GET", url)
            # Keep-alive reuses the one physical connection.
            assert _require(second.tls_handshake_result).connection_id == first_id
            # The first response keeps its own immutable snapshot.
            assert _require(first.tls_handshake_result).connection_id == first_id

    def test_response_snapshot_survives_connection_replacement(self) -> None:
        with tls_origin(alpn=["http/1.1"], close_after_response=True) as port:
            manager = urllib3.PoolManager(
                ca_certs=DEFAULT_CA,
                retries=False,
                tls_handshake_reporting=True,
            )
            url = f"https://localhost:{port}/"
            first = manager.request("GET", url)
            first_id = _require(first.tls_handshake_result).connection_id
            second = manager.request("GET", url)
            assert _require(second.tls_handshake_result).connection_id != first_id
            first_result = _require(first.tls_handshake_result)
            assert first_result.connection_id == first_id
            assert first_result.tls_completed is True

    def test_streaming_response_carries_snapshot(self) -> None:
        with tls_origin(alpn=["http/1.1"]) as port:
            manager = urllib3.PoolManager(
                ca_certs=DEFAULT_CA,
                retries=False,
                tls_handshake_reporting=True,
            )
            url = f"https://localhost:{port}/"
            response = manager.request("GET", url, preload_content=False)
            first_result = _require(response.tls_handshake_result)
            assert first_result.tls_completed is True
            first_id = first_result.connection_id
            assert response.read() == b"hi"
            response.release_conn()
            second = manager.request("GET", url, preload_content=False)
            assert _require(second.tls_handshake_result).connection_id == first_id
            assert second.read() == b"hi"
            second.release_conn()

    def test_disabled_by_default(self) -> None:
        with tls_origin(alpn=["http/1.1"]) as port:
            manager = urllib3.PoolManager(ca_certs=DEFAULT_CA, retries=False)
            response = manager.request("GET", f"https://localhost:{port}/")
            assert response.tls_handshake_result is None
            pool = manager.connection_from_host("localhost", port, "https")
            assert pool.get_tls_handshake_result() is None
            assert manager.get_tls_handshake_result() is None

    def test_url_query_does_not_create_pool_or_connect(self) -> None:
        manager = urllib3.PoolManager(tls_handshake_reporting=True)
        assert len(manager.pools) == 0
        assert (
            manager.get_tls_handshake_result("https://never-contacted.example/") is None
        )
        assert len(manager.pools) == 0

    def test_failure_result_queryable_after_pool_discards_connection(self) -> None:
        with closing_tcp_server() as port:
            with urllib3.HTTPSConnectionPool(
                "localhost", port, retries=0, tls_handshake_reporting=True
            ) as pool:
                with pytest.raises(MaxRetryError):
                    pool.request("GET", "/")
                result = pool.get_tls_handshake_result()
                assert result is not None
                assert result.failure_stage is TLSHandshakeStage.CLOSED
                assert result.closed is True

    def test_exception_types_unchanged_when_enabled(self) -> None:
        with tls_origin() as port:
            with urllib3.HTTPSConnectionPool(
                "localhost", port, tls_handshake_reporting=True
            ) as pool:
                with pytest.raises(MaxRetryError) as enabled_err:
                    pool.request("GET", "/")
        with tls_origin() as port:
            with urllib3.HTTPSConnectionPool("localhost", port) as pool:
                with pytest.raises(MaxRetryError) as disabled_err:
                    pool.request("GET", "/")
        assert type(enabled_err.value.reason) is type(disabled_err.value.reason)
        assert isinstance(enabled_err.value.reason, SSLError)


class TestTLSHandshakeProxies:
    def test_http_proxy_connect_success(self) -> None:
        with (
            tls_origin(alpn=["http/1.1"]) as origin_port,
            connect_proxy("http") as proxy_port,
        ):
            manager = urllib3.ProxyManager(
                f"http://127.0.0.1:{proxy_port}",
                ca_certs=DEFAULT_CA,
                retries=False,
                tls_handshake_reporting=True,
            )
            target_url = f"https://localhost:{origin_port}/"
            response = manager.request("GET", target_url)
            result = _require(response.tls_handshake_result)
            assert response.status == 200
            assert result.used_proxy is True
            assert result.proxy_scheme == "http"
            assert result.proxy_host == "127.0.0.1"
            assert result.proxy_tunnel_established is True
            _assert_success(result, host="localhost")
            assert result.sni == "localhost"
            proxy_hop = result.proxy_result
            assert isinstance(proxy_hop, TLSHandshakeResult)
            assert set(proxy_hop._fields) == set(RESULT_FIELDS)
            assert proxy_hop.target_host == "127.0.0.1"
            assert proxy_hop.tls_completed is False
            assert proxy_hop.cert_verified is False
            assert proxy_hop.proxy_tunnel_established is True
            assert proxy_hop.failure_stage is TLSHandshakeStage.NONE
            assert (
                _require(manager.get_tls_handshake_result(target_url)).connection_id
                == result.connection_id
            )

    def test_http_proxy_connect_refused(self) -> None:
        with (
            tls_origin(alpn=["http/1.1"]) as origin_port,
            connect_proxy("http", status=403) as proxy_port,
        ):
            manager = urllib3.ProxyManager(
                f"http://127.0.0.1:{proxy_port}",
                ca_certs=DEFAULT_CA,
                tls_handshake_reporting=True,
            )
            with pytest.raises(MaxRetryError) as err:
                manager.request("GET", f"https://localhost:{origin_port}/")
            assert isinstance(err.value.reason, ProxyError)
            result = _require(manager.get_tls_handshake_result())
            assert result.tls_completed is False
            assert result.proxy_tunnel_established is False
            assert result.failure_stage is TLSHandshakeStage.PROXY_CONNECT
            proxy_hop = result.proxy_result
            assert proxy_hop is not None
            assert proxy_hop.failure_stage is TLSHandshakeStage.PROXY_CONNECT
            assert proxy_hop.proxy_tunnel_established is False

    def test_https_proxy_connect_success(self) -> None:
        with (
            tls_origin(alpn=["http/1.1"]) as origin_port,
            connect_proxy("https") as proxy_port,
        ):
            manager = urllib3.ProxyManager(
                f"https://127.0.0.1:{proxy_port}",
                ca_certs=DEFAULT_CA,
                proxy_assert_hostname="localhost",
                retries=False,
                tls_handshake_reporting=True,
            )
            response = manager.request("GET", f"https://localhost:{origin_port}/")
            result = _require(response.tls_handshake_result)
            assert response.status == 200
            _assert_success(result, host="localhost")
            assert result.proxy_scheme == "https"
            assert result.proxy_tunnel_established is True
            proxy_hop = result.proxy_result
            assert proxy_hop is not None
            assert proxy_hop.tls_completed is True
            assert proxy_hop.cert_verified is True
            assert proxy_hop.tls_version in ("TLSv1.2", "TLSv1.3")
            assert proxy_hop.sni == "127.0.0.1"
            assert proxy_hop.proxy_tunnel_established is True
            assert proxy_hop.failure_stage is TLSHandshakeStage.NONE

    def test_https_proxy_connected_but_target_tls_fails(self) -> None:
        with (
            tls_origin(alpn=["http/1.1"]) as origin_port,
            connect_proxy("https") as proxy_port,
        ):
            proxy_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            proxy_context.load_verify_locations(DEFAULT_CA)
            manager = urllib3.ProxyManager(
                f"https://127.0.0.1:{proxy_port}",
                proxy_ssl_context=proxy_context,
                proxy_assert_hostname="localhost",
                tls_handshake_reporting=True,
            )
            with pytest.raises(MaxRetryError) as err:
                manager.request("GET", f"https://localhost:{origin_port}/")
            assert isinstance(err.value.reason, SSLError)
            result = _require(manager.get_tls_handshake_result())
            assert result.tls_completed is False
            assert result.failure_stage is TLSHandshakeStage.CERTIFICATE
            assert result.proxy_tunnel_established is True
            proxy_hop = result.proxy_result
            assert proxy_hop is not None
            assert proxy_hop.tls_completed is True
            assert proxy_hop.cert_verified is True
            assert proxy_hop.proxy_tunnel_established is True
            assert proxy_hop.failure_stage is TLSHandshakeStage.NONE

    def test_https_proxy_tls_failure(self) -> None:
        with (
            tls_origin(alpn=["http/1.1"]) as origin_port,
            connect_proxy("https") as proxy_port,
        ):
            manager = urllib3.ProxyManager(
                f"https://127.0.0.1:{proxy_port}",
                tls_handshake_reporting=True,
            )
            with pytest.raises(MaxRetryError) as err:
                manager.request("GET", f"https://localhost:{origin_port}/")
            assert isinstance(err.value.reason, ProxyError)
            result = _require(manager.get_tls_handshake_result())
            assert result.tls_completed is False
            assert result.failure_stage is TLSHandshakeStage.PROXY_TLS
            assert result.proxy_tunnel_established is False
            proxy_hop = result.proxy_result
            assert proxy_hop is not None
            assert proxy_hop.tls_completed is False
            assert proxy_hop.failure_stage is TLSHandshakeStage.CERTIFICATE

    def test_https_proxy_connect_refused(self) -> None:
        with (
            tls_origin(alpn=["http/1.1"]) as origin_port,
            connect_proxy("https", status=502) as proxy_port,
        ):
            manager = urllib3.ProxyManager(
                f"https://127.0.0.1:{proxy_port}",
                ca_certs=DEFAULT_CA,
                proxy_assert_hostname="localhost",
                tls_handshake_reporting=True,
            )
            with pytest.raises(MaxRetryError):
                manager.request("GET", f"https://localhost:{origin_port}/")
            result = _require(manager.get_tls_handshake_result())
            assert result.failure_stage is TLSHandshakeStage.PROXY_CONNECT
            proxy_hop = result.proxy_result
            assert proxy_hop is not None
            assert proxy_hop.tls_completed is True
            assert proxy_hop.proxy_tunnel_established is False
            assert proxy_hop.failure_stage is TLSHandshakeStage.PROXY_CONNECT


class TestTLSHandshakeALPN:
    def test_alpn_negotiated_and_offered(self) -> None:
        with tls_origin(alpn=["http/1.1"]) as port:
            conn = HTTPSConnection(
                "localhost", port, ca_certs=DEFAULT_CA, tls_handshake_reporting=True
            )
            conn.connect()
            result = _require(conn.tls_handshake_result)
            conn.close()
            assert result.alpn_offered == ("http/1.1",)
            assert result.alpn_negotiated == "http/1.1"
            assert result.alpn_mismatch is False

    def test_no_alpn_selection_is_reported_explicitly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The client offers a protocol the server doesn't speak. OpenSSL's
        # stdlib server continues the handshake without selecting anything;
        # the report must make that explicit instead of claiming success.
        monkeypatch.setattr(ssl_, "ALPN_PROTOCOLS", ["spdy/2"])
        with tls_origin(alpn=["http/1.1"]) as port:
            conn = HTTPSConnection(
                "localhost",
                port,
                cert_reqs="CERT_NONE",
                assert_hostname=False,
                tls_handshake_reporting=True,
            )
            conn.connect()
            result = _require(conn.tls_handshake_result)
            conn.close()
            assert result.alpn_offered == ("spdy/2",)
            assert result.alpn_negotiated is None
            assert result.alpn_mismatch is False
            assert result.tls_completed is True

    def test_unoffered_alpn_selection_is_mismatch(self) -> None:
        from urllib3.tls import _NegotiatedInfo, _result

        # Handshake info with a protocol outside the offered set is flagged.
        info = _NegotiatedInfo(
            version="TLSv1.3",
            cipher=("TLS_AES_128_GCM_SHA256", "TLSv1.3", 128),
            alpn_offered=("http/1.1",),
            alpn_negotiated="h2",
            alpn_mismatch=True,
            fingerprint="a" * 64,
        )
        result = _result(
            connection_id=1,
            target_host="localhost",
            target_port=443,
            sni="localhost",
            used_proxy=False,
            proxy_scheme=None,
            proxy_host=None,
            tunnel_established=False,
            tls_completed=True,
            cert_verified=True,
            failure_stage=TLSHandshakeStage.NONE,
            info=info,
        )
        assert result.alpn_mismatch is True
        assert result.alpn_negotiated == "h2"

    def test_alpn_abort_stage_plumbing(self) -> None:
        # Strict servers abort with no_application_protocol: the stage note
        # reported by the wrap layer must surface as failure_stage ALPN.
        from urllib3.tls import _ConnectTracker

        recorder = TLSHandshakeRecorder()
        tracker = _ConnectTracker(
            recorder,
            target_host="localhost",
            target_port=443,
            used_proxy=False,
            proxy_scheme=None,
            proxy_host=None,
            proxy_port=None,
        )
        err = ssl.SSLError(1, "NO_APPLICATION_PROTOCOL")
        err.reason = "NO_APPLICATION_PROTOCOL"
        tracker.stage_sink("target")(TLSHandshakeStage.ALPN, err, None)
        tracker.fail(err)
        result = tracker.result
        assert result is not None
        assert result.failure_stage is TLSHandshakeStage.ALPN
        assert result.tls_completed is False
        assert result.alpn_negotiated is None
        assert recorder.latest() is result


class TestTLSHandshakeDummyProxyServer:
    def test_full_path_via_dummy_https_proxy(
        self,
        san_proxy_with_server: tuple[ServerConfig, ServerConfig],
    ) -> None:
        proxy, server = san_proxy_with_server
        manager = urllib3.ProxyManager(
            f"https://{_hostport(proxy.host, proxy.port)}",
            ca_certs=server.ca_certs,
            tls_handshake_reporting=True,
        )
        response = manager.request(
            "GET", f"https://{_hostport(server.host, server.port)}/"
        )
        assert response.status == 200
        result = _require(response.tls_handshake_result)
        _assert_success(result, host=server.host)
        assert result.proxy_scheme == "https"
        assert result.proxy_tunnel_established is True
        proxy_hop = result.proxy_result
        assert proxy_hop is not None
        assert proxy_hop.tls_completed is True
        assert proxy_hop.proxy_tunnel_established is True
