from __future__ import annotations

import asyncio
import dataclasses
import enum
import threading
import time
import typing
from collections import deque
from collections.abc import Callable as _Callable
from collections.abc import Iterable as _Iterable
from http.client import HTTPException, RemoteDisconnected

from .exceptions import (
    NewConnectionError,
    ProtocolError,
    ProxyError,
    ReadTimeoutError,
    SSLError,
    TimeoutError,
)

__all__ = [
    "FaultState",
    "FaultReason",
    "FaultIsolationConfig",
    "FaultIsolationPolicy",
    "FaultSnapshot",
    "create_fault_isolation",
    "classify_connection_fault",
    "is_user_cancellation",
]


class FaultState(enum.Enum):
    """State of a :class:`FaultIsolationPolicy`."""

    #: Requests flow normally and failures are counted.
    CLOSED = "closed"
    #: New connection attempts are rejected until the recovery timeout elapses.
    OPEN = "open"
    #: Exactly one probe request is allowed to check whether the target recovered.
    HALF_OPEN = "half_open"


class FaultReason:
    """Reasons why a fault was recorded or why a pool entered the open state."""

    #: TCP connection establishment failed (refused, timeout, DNS, ...).
    CONNECT = "connect"
    #: Connection to the proxy (including HTTP CONNECT tunnelling) failed.
    PROXY_CONNECT = "proxy_connect"
    #: TLS handshake / certificate verification failed.
    TLS_HANDSHAKE = "tls_handshake"
    #: An established connection was aborted or closed by the peer.
    CONNECTION_CLOSED = "connection_closed"
    #: A configured HTTP response status code was received.
    HTTP_STATUS = "http_status"
    #: A response read error (only counted when explicitly enabled).
    READ_ERROR = "read_error"
    #: The request was cancelled by the caller (only counted when enabled).
    CANCELLED = "cancelled"


_TYPE_STATUS_MATCHER = typing.Union[
    _Iterable[int], _Callable[[int], bool]
]


@dataclasses.dataclass(frozen=True)
class FaultIsolationConfig:
    """
    Configuration for an optional connection fault isolation policy.

    The same configuration can safely be shared between many pools; each pool
    still keeps its own isolated fault state.

    :param failure_threshold:
        Number of failures tolerated before the target enters the open state.

    :param failure_window:
        If ``None`` (the default), failures are counted consecutively and any
        success resets the counter. If set to a number of seconds, failures are
        counted within a rolling time window: reaching ``failure_threshold``
        failures inside the window opens the state regardless of interleaved
        successes.

    :param recovery_timeout:
        Number of seconds spent in the open state, rejecting new connection
        attempts, before a single half-open probe request is allowed.

    :param count_statuses:
        A set of HTTP status codes (or a callable mapping a status code to a
        boolean) that should be counted as target failures. ``None`` (the
        default) means HTTP status codes never count.

    :param count_read_errors:
        If ``True``, response read errors such as read timeouts or truncated
        response bodies count as failures. They do not count by default.

    :param count_cancelled:
        If ``True``, requests aborted by the caller
        (``KeyboardInterrupt`` / ``asyncio.CancelledError``) count as failures.
        They do not count by default.
    """

    failure_threshold: int = 5
    failure_window: float | None = None
    recovery_timeout: float = 30.0
    count_statuses: _TYPE_STATUS_MATCHER | None = None
    count_read_errors: bool = False
    count_cancelled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.failure_threshold, int) or isinstance(
            self.failure_threshold, bool
        ):
            raise TypeError("failure_threshold must be an int")
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.failure_window is not None and self.failure_window <= 0:
            raise ValueError("failure_window must be > 0 or None")
        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")
        statuses = self.count_statuses
        if statuses is not None and not callable(statuses):
            if isinstance(statuses, (str, bytes)) or not isinstance(
                statuses, _Iterable
            ):
                raise TypeError(
                    "count_statuses must be a set/iterable of ints, "
                    "a callable, or None"
                )
            statuses = frozenset(statuses)
            object.__setattr__(self, "count_statuses", statuses)

    def status_is_counted(self, status_code: int) -> bool:
        """Return whether an HTTP response status code should count as a fault."""
        matcher = self.count_statuses
        if matcher is None:
            return False
        if callable(matcher):
            return bool(matcher(status_code))
        return status_code in matcher


@dataclasses.dataclass(frozen=True)
class FaultSnapshot:
    """
    Immutable point-in-time view of a :class:`FaultIsolationPolicy`.

    :param state: Current state (``closed``/``open``/``half_open``).
    :param reason: Reason for the most recent opening of the state, if any.
    :param consecutive_failures: Failures since the last successful connection.
    :param window_failures: Failures counted inside the rolling window.
    :param total_failures: Total failures observed by this policy.
    :param opened_at: Clock value at which the state was opened.
    :param remaining_recovery_time: Seconds left before a probe is permitted.
    :param probe_in_flight: Whether the single half-open probe is occupied.
    :param last_error: String representation of the last failure.
    :param last_failure_at: Clock value of the last recorded failure.
    :param recent_failures: Recent ``(clock, reason)`` failure records.
    :param recovery_succeeded: Outcome of the last recovery probe (``None`` if
        no recovery has completed yet).
    :param recovered_at: Clock value of the last completed recovery probe.
    :param recovery_error: String representation of the last failed probe error.
    """

    state: FaultState
    reason: str | None
    consecutive_failures: int
    window_failures: int
    total_failures: int
    opened_at: float | None
    remaining_recovery_time: float
    probe_in_flight: bool
    last_error: str | None
    last_failure_at: float | None
    recent_failures: tuple[tuple[float, str], ...]
    recovery_succeeded: bool | None
    recovered_at: float | None
    recovery_error: str | None


class _FaultIsolationOpen(Exception):
    """Internal signal raised while the target is isolated.

    The connection pool layer converts this into the public
    :class:`urllib3.exceptions.FaultIsolationError`.
    """

    def __init__(
        self,
        reason: str | None,
        remaining: float | None,
        probe_in_flight: bool,
    ) -> None:
        super().__init__(reason, remaining, probe_in_flight)
        self.reason = reason
        self.remaining = remaining
        self.probe_in_flight = probe_in_flight


_FAILURE_RECORD = tuple[float, str]


class FaultIsolationPolicy:
    """
    Thread-safe connection fault isolation policy (circuit breaker).

    A policy tracks connection-level failures of one connection pool. While
    the failure threshold is not reached the state is :attr:`FaultState.CLOSED`
    and every request is admitted. Once the threshold is reached the state
    becomes :attr:`FaultState.OPEN`: new connection attempts are rejected for
    ``recovery_timeout`` seconds. When that timeout elapses the state becomes
    :attr:`FaultState.HALF_OPEN` and exactly one probe request is admitted; a
    successful connection atomically clears the fault state while a failed
    probe reopens it without releasing any concurrent request.
    """

    def __init__(
        self,
        config: FaultIsolationConfig | None = None,
        *,
        failure_threshold: int = 5,
        failure_window: float | None = None,
        recovery_timeout: float = 30.0,
        count_statuses: _TYPE_STATUS_MATCHER | None = None,
        count_read_errors: bool = False,
        count_cancelled: bool = False,
        _time_func: typing.Callable[[], float] = time.monotonic,
    ) -> None:
        if config is None:
            config = FaultIsolationConfig(
                failure_threshold=failure_threshold,
                failure_window=failure_window,
                recovery_timeout=recovery_timeout,
                count_statuses=count_statuses,
                count_read_errors=count_read_errors,
                count_cancelled=count_cancelled,
            )
        self.config: typing.Final[FaultIsolationConfig] = config
        self._time = _time_func
        self._lock = threading.RLock()
        self._state = FaultState.CLOSED
        self._consecutive_failures = 0
        self._total_failures = 0
        self._failures: deque[_FAILURE_RECORD] = deque()
        self._opened_at: float | None = None
        self._open_reason: str | None = None
        self._probe_in_flight = False
        self._last_error: str | None = None
        self._last_failure_at: float | None = None
        self._recovery_succeeded: bool | None = None
        self._recovered_at: float | None = None
        self._recovery_error: str | None = None

    @classmethod
    def from_config(
        cls,
        config: FaultIsolationConfig,
        *,
        _time_func: typing.Callable[[], float] = time.monotonic,
    ) -> FaultIsolationPolicy:
        return cls(config, _time_func=_time_func)

    @property
    def state(self) -> FaultState:
        with self._lock:
            return self._state

    @property
    def probe_in_flight(self) -> bool:
        with self._lock:
            return self._probe_in_flight

    def acquire(self) -> FaultPermit:
        """
        Acquire admission for one request attempt.

        Returns a :class:`FaultPermit` for a normal request or for the single
        half-open probe. Raises :class:`_FaultIsolationOpen` while new
        connection attempts must be rejected.
        """
        with self._lock:
            now = self._time()
            if self._state is FaultState.CLOSED:
                return FaultPermit(self, is_probe=False)

            if self._state is FaultState.OPEN:
                if (
                    self._opened_at is not None
                    and now - self._opened_at >= self.config.recovery_timeout
                ):
                    self._state = FaultState.HALF_OPEN
                    self._probe_in_flight = True
                    return FaultPermit(self, is_probe=True)
                raise _FaultIsolationOpen(
                    reason=self._open_reason,
                    remaining=self._remaining_recovery_time(now),
                    probe_in_flight=False,
                )

            # HALF_OPEN: a single probe owns the state until it settles.
            if not self._probe_in_flight:
                self._probe_in_flight = True
                return FaultPermit(self, is_probe=True)
            raise _FaultIsolationOpen(
                reason=self._open_reason, remaining=None, probe_in_flight=True
            )

    def reset(self) -> None:
        """Clear every failure record and return to the closed state."""
        with self._lock:
            self._state = FaultState.CLOSED
            self._consecutive_failures = 0
            self._total_failures = 0
            self._failures.clear()
            self._opened_at = None
            self._open_reason = None
            self._probe_in_flight = False
            self._last_error = None
            self._last_failure_at = None
            self._recovery_succeeded = None
            self._recovered_at = None
            self._recovery_error = None

    def snapshot(self) -> FaultSnapshot:
        """Return an immutable :class:`FaultSnapshot` of the current state."""
        with self._lock:
            now = self._time()
            self._prune_window(now)
            return FaultSnapshot(
                state=self._state,
                reason=self._open_reason,
                consecutive_failures=self._consecutive_failures,
                window_failures=len(self._failures),
                total_failures=self._total_failures,
                opened_at=self._opened_at,
                remaining_recovery_time=self._remaining_recovery_time(now),
                probe_in_flight=self._probe_in_flight,
                last_error=self._last_error,
                last_failure_at=self._last_failure_at,
                recent_failures=tuple(self._failures),
                recovery_succeeded=self._recovery_succeeded,
                recovered_at=self._recovered_at,
                recovery_error=self._recovery_error,
            )

    def _remaining_recovery_time(self, now: float) -> float:
        if self._state is not FaultState.OPEN or self._opened_at is None:
            return 0.0
        return max(
            0.0,
            self.config.recovery_timeout - (now - self._opened_at),
        )

    def _prune_window(self, now: float) -> None:
        window = self.config.failure_window
        if window is None:
            return
        cutoff = now - window
        while self._failures and self._failures[0][0] < cutoff:
            self._failures.popleft()

    def _record_failure(
        self,
        permit: FaultPermit,
        reason: str,
        error: BaseException | str,
    ) -> None:
        with self._lock:
            if permit._settled:
                return
            permit._settled = True

            now = self._time()
            self._total_failures += 1
            self._consecutive_failures += 1
            self._open_reason = reason
            self._last_error = _error_repr(error)
            self._last_failure_at = now

            self._failures.append((now, reason))
            if self.config.failure_window is None:
                # Only the most recent failures matter for consecutive counting.
                while len(self._failures) > self.config.failure_threshold:
                    self._failures.popleft()
            else:
                self._prune_window(now)

            if permit.is_probe:
                self._probe_in_flight = False
                if self._state is not FaultState.CLOSED:
                    # A failed (or stale) probe reopens the state with a fresh
                    # recovery timeout; concurrent requests stay rejected.
                    self._state = FaultState.OPEN
                    self._opened_at = now
                    self._recovery_succeeded = False
                    self._recovered_at = now
                    self._recovery_error = self._last_error
                    return

            if self._state is FaultState.CLOSED and self._threshold_reached(now):
                self._state = FaultState.OPEN
                self._opened_at = now

    def _record_success(self, permit: FaultPermit) -> None:
        with self._lock:
            if permit._settled:
                return
            permit._settled = True

            now = self._time()
            recovering = self._state is not FaultState.CLOSED
            if recovering:
                self._recovery_succeeded = True
                self._recovered_at = now
                self._recovery_error = None

            # Successful connection atomically clears the fault state.
            self._state = FaultState.CLOSED
            self._opened_at = None
            self._open_reason = None
            self._probe_in_flight = False
            self._consecutive_failures = 0
            if recovering or self.config.failure_window is None:
                # A verified recovery (or a success in consecutive mode)
                # wipes the history. In rolling-window mode ordinary successes
                # leave recent failures in place until the window expires.
                self._failures.clear()
            else:
                self._prune_window(now)

    def _abandon(self, permit: FaultPermit) -> None:
        """Settle a permit without a verdict (e.g. empty pool, cancellation)."""
        with self._lock:
            if permit._settled:
                return
            permit._settled = True
            if permit.is_probe and self._state is FaultState.HALF_OPEN:
                # The probe produced no usable verdict: stay conservative,
                # reopen with a fresh timeout and keep rejecting other requests.
                self._state = FaultState.OPEN
                self._opened_at = self._time()
                self._probe_in_flight = False

    def _threshold_reached(self, now: float) -> bool:
        if self.config.failure_window is not None:
            self._prune_window(now)
            return len(self._failures) >= self.config.failure_threshold
        return self._consecutive_failures >= self.config.failure_threshold


class FaultPermit:
    """
    Admission granted by :meth:`FaultIsolationPolicy.acquire`.

    Exactly one of :meth:`record_success`, :meth:`record_failure`,
    :meth:`record_status` or :meth:`abandon` settles the permit; subsequent
    calls are ignored so a single attempt can never be counted twice. The
    connection pool layer calls :meth:`release` in a ``finally`` block, which
    settles an unsettled probe neutrally.
    """

    __slots__ = ("_policy", "is_probe", "_settled")

    def __init__(self, policy: FaultIsolationPolicy, *, is_probe: bool) -> None:
        self._policy = policy
        self.is_probe = is_probe
        self._settled = False

    def record_success(self) -> None:
        """Record a successful connection and clear the fault state."""
        self._policy._record_success(self)

    def record_failure(self, reason: str, error: BaseException | str) -> None:
        """Record a connection failure of the given ``reason``."""
        self._policy._record_failure(self, reason, error)

    def record_status(self, status_code: int) -> None:
        """Record an HTTP response status, counting it when configured."""
        if self._policy.config.status_is_counted(status_code):
            self._policy._record_failure(
                self, FaultReason.HTTP_STATUS, f"HTTP {status_code}"
            )
        else:
            self._policy._record_success(self)

    def abandon(self) -> None:
        """Release the permit without recording success or failure."""
        self._policy._abandon(self)

    def release(self) -> None:
        """Idempotent cleanup used by the pool's ``finally`` block."""
        if not self._settled:
            self._policy._abandon(self)


def _error_repr(error: BaseException | str) -> str:
    if isinstance(error, BaseException):
        return f"{type(error).__name__}: {error}"
    return str(error)


def is_user_cancellation(error: BaseException) -> bool:
    """Return whether ``error`` represents a caller-initiated cancellation."""
    return isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))


def classify_connection_fault(
    original_error: BaseException,
    wrapped_error: BaseException,
    config: FaultIsolationConfig,
) -> str | None:
    """
    Classify a failure caught by ``HTTPConnectionPool.urlopen()`` into one of
    the :class:`FaultReason` values, or ``None`` when the failure must not be
    counted according to ``config``.

    :param original_error: The exception as caught by the pool.
    :param wrapped_error: The exception after the pool's proxy/protocol wrapping.
    :param config: The policy configuration governing optional categories.
    """
    if isinstance(wrapped_error, ProxyError):
        return FaultReason.PROXY_CONNECT

    if isinstance(wrapped_error, SSLError):
        return FaultReason.TLS_HANDSHAKE

    if isinstance(original_error, ReadTimeoutError):
        return FaultReason.READ_ERROR if config.count_read_errors else None

    if isinstance(wrapped_error, (NewConnectionError, TimeoutError)):
        return FaultReason.CONNECT

    if isinstance(wrapped_error, ProtocolError):
        # ``OSError`` (including ``http.client.RemoteDisconnected``) means the
        # established connection was aborted/passively closed by the peer.
        if isinstance(original_error, OSError) or isinstance(
            original_error, RemoteDisconnected
        ):
            return FaultReason.CONNECTION_CLOSED
        if isinstance(original_error, HTTPException):
            return FaultReason.READ_ERROR if config.count_read_errors else None

    return None


_FAULT_ISOLATION_ARG = typing.Union[FaultIsolationConfig, FaultIsolationPolicy, bool]


def create_fault_isolation(
    value: _FAULT_ISOLATION_ARG | None,
) -> FaultIsolationPolicy | None:
    """
    Build an independent :class:`FaultIsolationPolicy` from a pool/manager
    argument: ``None``/``False`` disable fault isolation, ``True`` enables it
    with the default configuration, a config yields a fresh policy, and an
    existing policy is returned unchanged.
    """
    if value is None or value is False:
        return None
    if isinstance(value, FaultIsolationPolicy):
        return value
    if isinstance(value, FaultIsolationConfig):
        return FaultIsolationPolicy(value)
    if value is True:
        return FaultIsolationPolicy()
    raise TypeError(
        "fault_isolation must be None, a bool, a FaultIsolationConfig, "
        "or a FaultIsolationPolicy"
    )
