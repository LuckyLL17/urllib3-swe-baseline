"""
Optional TLS handshake result reporting.

This module is opt-in. Nothing in here performs network I/O and nothing in
here modifies a user-supplied :class:`ssl.SSLContext`: negotiated values are
read locally from an already connected socket, certificates are only ever
hashed (never stored or exposed), and every helper is defensive so that a
reporting failure can never alter the underlying connection or request flow.
"""

from __future__ import annotations

import enum
import hashlib
import itertools
import threading
import typing

__all__ = [
    "TLSHandshakeStage",
    "TLSHandshakeResult",
    "TLSHandshakeRecorder",
]


class TLSHandshakeStage(str, enum.Enum):
    """
    Stable identifiers for the stage at which a connection attempt ended.

    ``NONE`` means no failure was recorded (the reported state itself still
    says whether the TLS handshake actually completed).
    """

    NONE = "none"
    #: Establishing the raw TCP socket failed (DNS, timeout, refused, ...).
    TCP_CONNECT = "tcp_connect"
    #: The TLS handshake to an HTTPS proxy failed; ``proxy_result`` on the
    #: reported result carries the proxy-hop specific failure stage.
    PROXY_TLS = "proxy_tls"
    #: The HTTP CONNECT exchange that should establish the tunnel failed.
    PROXY_CONNECT = "proxy_connect"
    #: The TLS handshake to the target failed without a more specific stage.
    TLS = "tls"
    #: Certificate chain verification failed.
    CERTIFICATE = "certificate"
    #: The certificate did not match the expected hostname.
    HOSTNAME = "hostname"
    #: ALPN negotiation failed or selected an unoffered protocol.
    ALPN = "alpn"
    #: The peer closed the connection (or reset it) during the handshake.
    CLOSED = "closed"
    #: The failure could not be classified.
    UNKNOWN = "unknown"


class TLSHandshakeResult(typing.NamedTuple):
    """
    Immutable description of one HTTPS connection attempt.

    The exact same structure is used for direct target TLS, for the TLS hop
    to an HTTPS proxy, and for a plaintext HTTP proxy CONNECT hop (where the
    TLS fields are ``None``/empty and ``tls_completed`` is ``False``).
    """

    #: Process-wide stable identifier for this physical connection attempt.
    connection_id: int
    #: Host this connection ultimately talks to (the tunnel target when a
    #: CONNECT tunnel is in use, otherwise the socket peer).
    target_host: str
    target_port: int | None
    #: SNI value actually offered during the TLS handshake.
    sni: str | None
    #: Whether the connection goes through any proxy.
    used_proxy: bool
    proxy_scheme: str | None
    proxy_host: str | None
    #: Whether an HTTP CONNECT tunnel was successfully established.
    proxy_tunnel_established: bool
    #: Whether the TLS handshake completed. When ``False`` no negotiated
    #: values, verification success or fingerprint are ever reported.
    tls_completed: bool
    #: Whether certificate verification (chain, hostname or fingerprint)
    #: passed according to urllib3's policy.
    cert_verified: bool
    tls_version: str | None
    #: ``(cipher_name, protocol_version, secret_bits)`` as returned by
    #: :meth:`ssl.SSLSocket.cipher`.
    cipher: tuple[str, str, int] | None
    alpn_offered: tuple[str, ...]
    alpn_negotiated: str | None
    #: ``True`` when the peer selected an ALPN protocol that was not offered.
    alpn_mismatch: bool
    #: SHA-256 fingerprint (hex) of the peer certificate. The raw or parsed
    #: certificate is intentionally never exposed.
    peer_cert_fingerprint: str | None
    failure_stage: TLSHandshakeStage
    error_type: str | None
    error_message: str | None
    #: Whether the connection was closed after this result was produced.
    closed: bool
    #: Result of the proxy hop (TLS to an HTTPS proxy or plaintext HTTP
    #: CONNECT), using this same structure.
    proxy_result: TLSHandshakeResult | None


_connection_id_counter = itertools.count(1)

_MAX_ERROR_MESSAGE = 500


def _new_connection_id() -> int:
    return next(_connection_id_counter)


def _error_type(exc: BaseException | None) -> str | None:
    if exc is None:
        return None
    exc_type = type(exc)
    name = f"{exc_type.__module__}.{exc_type.__qualname__}"
    if name.startswith("builtins."):
        name = name[len("builtins.") :]
    return name


def _error_message(exc: BaseException | None) -> str | None:
    if exc is None:
        return None
    try:
        message = str(exc)
    except Exception:
        return None
    if len(message) > _MAX_ERROR_MESSAGE:
        message = message[:_MAX_ERROR_MESSAGE] + "..."
    return message


class _NegotiatedInfo(typing.NamedTuple):
    version: str | None
    cipher: tuple[str, str, int] | None
    alpn_offered: tuple[str, ...]
    alpn_negotiated: str | None
    alpn_mismatch: bool
    fingerprint: str | None


def _try(method: typing.Callable[[], _R]) -> _R | None:
    try:
        return method()
    except Exception:
        return None


_R = typing.TypeVar("_R")


def _read_negotiated_info(sock: typing.Any) -> _NegotiatedInfo | None:
    """Read negotiated TLS values from a connected socket. Local calls only."""
    try:
        version = _try(lambda: sock.version())
        if not isinstance(version, str):
            version = None

        cipher = _try(lambda: sock.cipher())
        if not (
            isinstance(cipher, tuple)
            and len(cipher) == 3
            and isinstance(cipher[0], str)
        ):
            cipher = None

        negotiated = _try(lambda: sock.selected_alpn_protocol())
        if not isinstance(negotiated, str):
            negotiated = None

        offered: tuple[str, ...] = ()
        context = getattr(sock, "context", None)
        context_offered = _try(lambda: tuple(getattr(context, "alpn_protocols", ())))
        if context_offered:
            offered = typing.cast(tuple[str, ...], context_offered)
        else:
            # Fall back to the list urllib3 configures on every context.
            try:
                from .util import ssl_

                offered = tuple(ssl_.ALPN_PROTOCOLS)
            except Exception:
                offered = ()

        mismatch = (
            bool(offered) and negotiated is not None and negotiated not in offered
        )

        fingerprint: str | None = None
        der_cert = _try(lambda: sock.getpeercert(binary_form=True))
        if isinstance(der_cert, (bytes, bytearray)) and der_cert:
            fingerprint = hashlib.sha256(bytes(der_cert)).hexdigest()

        return _NegotiatedInfo(
            version=version,
            cipher=typing.cast("tuple[str, str, int] | None", cipher),
            alpn_offered=offered,
            alpn_negotiated=negotiated,
            alpn_mismatch=mismatch,
            fingerprint=fingerprint,
        )
    except Exception:
        return None


def classify_failure_stage(exc: BaseException | None) -> TLSHandshakeStage:
    """
    Best-effort classification of an exception raised while connecting or
    handshaking. Never raises.
    """
    try:
        # urllib3's own hostname mismatch exception, kept distinct even if it
        # also arrives through the generic TLS error wrapper.
        from .util.ssl_match_hostname import CertificateError

        if isinstance(exc, CertificateError):
            return TLSHandshakeStage.HOSTNAME
    except Exception:
        pass

    reason = ""
    reason_attr = getattr(exc, "reason", None)
    if isinstance(reason_attr, str):
        reason = reason_attr
    try:
        reason = f"{reason} {exc}"
    except Exception:
        pass
    reason_upper = reason.upper()

    if (
        "CERTIFICATE_VERIFY_FAILED" in reason_upper
        or "CERTIFICATE_VERIFY" in reason_upper
        or "UNABLE TO GET LOCAL ISSUER CERTIFICATE" in reason_upper
        or "SELF SIGNED CERTIFICATE" in reason_upper
        or "SELF-SIGNED" in reason_upper
    ):
        return TLSHandshakeStage.CERTIFICATE
    if "NO_APPLICATION_PROTOCOL" in reason_upper or "ALPN" in reason_upper:
        return TLSHandshakeStage.ALPN
    if (
        "UNEXPECTED_EOF_WHILE_READING" in reason_upper
        or "EOF_WAS_OBSERVED" in reason_upper
        or "UNEXPECTED EOF" in reason_upper
        or "EOF OCCURRED" in reason_upper
        or "TLSV1_ALERT_RECORD_OVERFLOW" in reason_upper
        or "CONNECTION_RESET" in reason_upper
        or "CONNECTION_ABORTED" in reason_upper
        or "BROKEN PIPE" in reason_upper
        or "SHUTDOWN_WHILE_READING" in reason_upper
    ):
        return TLSHandshakeStage.CLOSED
    try:
        import ssl

        # Any other SSL-layer failure happened during the TLS handshake.
        if isinstance(exc, ssl.SSLError):
            return TLSHandshakeStage.TLS
    except Exception:
        pass
    if isinstance(exc, OSError):
        # Raw socket errors surfaced during a handshake mean the peer went
        # away rather than completing TLS.
        errno = getattr(exc, "errno", None)
        if errno in (54, 104, 32, 103) or "EOF" in reason_upper:
            return TLSHandshakeStage.CLOSED
    return TLSHandshakeStage.UNKNOWN


class TLSHandshakeRecorder:
    """
    Holds the latest :class:`TLSHandshakeResult` observed by a connection,
    pool or pool manager.

    All operations are local (a short lock); queries never initiate network
    activity.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: TLSHandshakeResult | None = None

    def new_connection_id(self) -> int:
        return _new_connection_id()

    def record(self, result: TLSHandshakeResult) -> TLSHandshakeResult:
        with self._lock:
            self._latest = result
        return result

    def latest(self) -> TLSHandshakeResult | None:
        with self._lock:
            return self._latest


def _result(
    *,
    connection_id: int,
    target_host: str,
    target_port: int | None,
    sni: str | None,
    used_proxy: bool,
    proxy_scheme: str | None,
    proxy_host: str | None,
    tunnel_established: bool,
    tls_completed: bool,
    cert_verified: bool,
    failure_stage: TLSHandshakeStage,
    error: BaseException | None = None,
    closed: bool = False,
    info: _NegotiatedInfo | None = None,
    proxy_result: TLSHandshakeResult | None = None,
) -> TLSHandshakeResult:
    """Build a result, enforcing the "no incomplete handshake success" rule."""
    if not tls_completed:
        cert_verified = False
        tls_version = None
        cipher = None
        alpn_offered: tuple[str, ...] = ()
        alpn_negotiated = None
        alpn_mismatch = False
        fingerprint = None
    else:
        tls_version = info.version if info is not None else None
        cipher = info.cipher if info is not None else None
        alpn_offered = info.alpn_offered if info is not None else ()
        alpn_negotiated = info.alpn_negotiated if info is not None else None
        alpn_mismatch = info.alpn_mismatch if info is not None else False
        fingerprint = info.fingerprint if info is not None else None

    return TLSHandshakeResult(
        connection_id=connection_id,
        target_host=target_host,
        target_port=target_port,
        sni=sni,
        used_proxy=used_proxy,
        proxy_scheme=proxy_scheme,
        proxy_host=proxy_host,
        proxy_tunnel_established=tunnel_established,
        tls_completed=tls_completed,
        cert_verified=cert_verified,
        tls_version=tls_version,
        cipher=cipher,
        alpn_offered=alpn_offered,
        alpn_negotiated=alpn_negotiated,
        alpn_mismatch=alpn_mismatch,
        peer_cert_fingerprint=fingerprint,
        failure_stage=failure_stage,
        error_type=_error_type(error),
        error_message=_error_message(error),
        closed=closed,
        proxy_result=proxy_result,
    )


def snapshot_ssl_context(context: typing.Any) -> tuple[typing.Any, ...] | None:
    """
    Capture the user-settable policy attributes of a user-supplied
    SSLContext so they can be restored after urllib3's connect-time
    configuration. Never raises.
    """
    try:
        verify_mode = context.verify_mode
        check_hostname = bool(context.check_hostname)
        # Some SSLContext implementations (CPython 3.14+, pyOpenSSL) don't
        # expose an ALPN getter. Signal that with None so restoration leaves
        # the ALPN configuration untouched rather than wiping it.
        try:
            alpn_protocols: tuple[str, ...] | None = tuple(context.alpn_protocols or ())
        except AttributeError:
            alpn_protocols = None
        return (verify_mode, check_hostname, alpn_protocols)
    except Exception:
        return None


def restore_ssl_context(
    context: typing.Any, snapshot: tuple[typing.Any, ...] | None
) -> None:
    """
    Restore SSLContext attributes captured by :func:`snapshot_ssl_context`.
    The ordering respects ssl's rule that check_hostname=True requires
    CERT_REQUIRED. Never raises: reporting must not break a connection.
    """
    if snapshot is None:
        return
    try:
        verify_mode, check_hostname, alpn_protocols = snapshot
        if alpn_protocols is not None:
            try:
                if tuple(getattr(context, "alpn_protocols", ()) or ()) != tuple(
                    alpn_protocols
                ):
                    context.set_alpn_protocols(list(alpn_protocols))
            except Exception:
                pass
        try:
            if check_hostname:
                import ssl

                context.verify_mode = ssl.CERT_REQUIRED
                context.check_hostname = True
            else:
                context.check_hostname = False
                context.verify_mode = verify_mode
        except Exception:
            pass
    except Exception:
        pass


class _ConnectTracker:
    """
    Per-``connect()`` reporting state. Only instantiated when TLS handshake
    reporting is enabled; every method is failure-proof by construction.
    """

    def __init__(
        self,
        recorder: TLSHandshakeRecorder,
        *,
        target_host: str,
        target_port: int | None,
        used_proxy: bool,
        proxy_scheme: str | None,
        proxy_host: str | None,
        proxy_port: int | None,
    ) -> None:
        self.recorder = recorder
        self.connection_id = recorder.new_connection_id()
        self.target_host = target_host
        self.target_port = target_port
        self.used_proxy = used_proxy
        self.proxy_scheme = proxy_scheme
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port

        self.tunnel_established = False
        self.proxy_result: TLSHandshakeResult | None = None
        self.target_sni: str | None = None

        self._proxy_sni: str | None = None
        self._proxy_stage: TLSHandshakeStage | None = None
        self._proxy_info: _NegotiatedInfo | None = None
        self._target_stage: TLSHandshakeStage | None = None
        self._target_info: _NegotiatedInfo | None = None
        self._committed = False
        self.result: TLSHandshakeResult | None = None

    def stage_sink(self, hop: str) -> typing.Callable[..., None]:
        def note(
            stage: TLSHandshakeStage,
            exc: BaseException | None = None,
            sock: typing.Any = None,
        ) -> None:
            try:
                info = _read_negotiated_info(sock) if sock is not None else None
                if hop == "proxy":
                    self._proxy_stage = stage
                    self._proxy_info = info
                else:
                    self._target_stage = stage
                    self._target_info = info
            except Exception:
                pass

        return note

    def _build_proxy_hop(
        self,
        *,
        tls_completed: bool,
        cert_verified: bool,
        tunnel_established: bool,
        failure_stage: TLSHandshakeStage,
        error: BaseException | None = None,
        info: _NegotiatedInfo | None = None,
    ) -> TLSHandshakeResult:
        return _result(
            connection_id=self.connection_id,
            target_host=self.proxy_host or "",
            target_port=self.proxy_port,
            sni=self._proxy_sni if tls_completed else None,
            used_proxy=False,
            proxy_scheme=None,
            proxy_host=None,
            tunnel_established=tunnel_established,
            tls_completed=tls_completed,
            cert_verified=cert_verified,
            failure_stage=failure_stage,
            error=error,
            info=info,
            proxy_result=None,
        )

    def proxy_tls_succeeded(
        self,
        sock: typing.Any,
        *,
        sni: str,
        cert_verified: bool,
    ) -> None:
        try:
            self._proxy_sni = sni
            info = _read_negotiated_info(sock)
            self.proxy_result = self._build_proxy_hop(
                tls_completed=True,
                cert_verified=cert_verified,
                tunnel_established=False,
                failure_stage=TLSHandshakeStage.NONE,
                info=info,
            )
        except Exception:
            pass

    def proxy_tls_failed(self, exc: BaseException, *, sni: str) -> None:
        try:
            self._proxy_sni = sni
            stage = self._proxy_stage or classify_failure_stage(exc)
            self.proxy_result = self._build_proxy_hop(
                tls_completed=self._proxy_info is not None,
                cert_verified=False,
                tunnel_established=False,
                failure_stage=stage,
                error=exc,
                info=self._proxy_info,
            )
        except Exception:
            pass

    def proxy_connect_failed(self, exc: BaseException) -> None:
        try:
            if self.proxy_result is not None:
                # HTTPS proxy: the proxy TLS hop was up but CONNECT failed.
                self.proxy_result = self.proxy_result._replace(
                    proxy_tunnel_established=False,
                    failure_stage=TLSHandshakeStage.PROXY_CONNECT,
                    error_type=_error_type(exc),
                    error_message=_error_message(exc),
                )
            else:
                # Plain HTTP proxy: no proxy TLS at all.
                self.proxy_result = self._build_proxy_hop(
                    tls_completed=False,
                    cert_verified=False,
                    tunnel_established=False,
                    failure_stage=TLSHandshakeStage.PROXY_CONNECT,
                    error=exc,
                )
        except Exception:
            pass

    def tunnel_succeeded(self) -> None:
        try:
            self.tunnel_established = True
            if self.proxy_result is not None:
                self.proxy_result = self.proxy_result._replace(
                    proxy_tunnel_established=True,
                )
            else:
                # Plaintext HTTP proxy CONNECT succeeded: no TLS on the hop.
                self.proxy_result = self._build_proxy_hop(
                    tls_completed=False,
                    cert_verified=False,
                    tunnel_established=True,
                    failure_stage=TLSHandshakeStage.NONE,
                )
        except Exception:
            pass

    def _commit(self, result: TLSHandshakeResult) -> None:
        self._committed = True
        self.result = self.recorder.record(result)

    def target_succeeded(
        self,
        sock: typing.Any,
        *,
        sni: str,
        cert_verified: bool,
    ) -> None:
        try:
            self.target_sni = sni
            info = _read_negotiated_info(sock)
            result = _result(
                connection_id=self.connection_id,
                target_host=self.target_host,
                target_port=self.target_port,
                sni=sni,
                used_proxy=self.used_proxy,
                proxy_scheme=self.proxy_scheme,
                proxy_host=self.proxy_host,
                tunnel_established=self.tunnel_established,
                tls_completed=True,
                cert_verified=cert_verified,
                failure_stage=TLSHandshakeStage.NONE,
                info=info,
                proxy_result=self.proxy_result,
            )
            self._commit(result)
        except Exception:
            pass

    def forwarding_succeeded(
        self,
        sock: typing.Any,
        *,
        sni: str,
        cert_verified: bool,
    ) -> None:
        # A forwarding proxy is the TLS endpoint: the top-level result
        # describes the actual TLS connection to the proxy.
        try:
            self.target_sni = sni
            info = _read_negotiated_info(sock)
            result = _result(
                connection_id=self.connection_id,
                target_host=self.proxy_host or self.target_host,
                target_port=self.proxy_port or self.target_port,
                sni=sni,
                used_proxy=True,
                proxy_scheme=self.proxy_scheme,
                proxy_host=self.proxy_host,
                tunnel_established=False,
                tls_completed=True,
                cert_verified=cert_verified,
                failure_stage=TLSHandshakeStage.NONE,
                info=info,
                proxy_result=None,
            )
            self._commit(result)
        except Exception:
            pass

    def fail(
        self,
        exc: BaseException,
        stage: TLSHandshakeStage | None = None,
    ) -> None:
        try:
            if self._committed:
                return
            if stage is None:
                stage = self._target_stage or classify_failure_stage(exc)
            posthandshake = self._target_info is not None
            result = _result(
                connection_id=self.connection_id,
                target_host=self.target_host,
                target_port=self.target_port,
                sni=self.target_sni,
                used_proxy=self.used_proxy,
                proxy_scheme=self.proxy_scheme,
                proxy_host=self.proxy_host,
                tunnel_established=self.tunnel_established,
                tls_completed=posthandshake,
                cert_verified=False,
                failure_stage=stage,
                error=exc,
                info=self._target_info,
                proxy_result=self.proxy_result,
            )
            self._commit(result)
        except Exception:
            pass
