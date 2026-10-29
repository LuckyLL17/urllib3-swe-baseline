from __future__ import annotations

import asyncio
import ssl
import threading
import time
import typing
from http.client import IncompleteRead, RemoteDisconnected
from unittest.mock import Mock, patch

import pytest

from urllib3 import (
    FaultIsolationConfig,
    FaultIsolationPolicy,
    FaultReason,
    FaultSnapshot,
    FaultState,
    HTTPConnectionPool,
    HTTPSConnectionPool,
    PoolManager,
    ProxyManager,
    Retry,
)
from urllib3.exceptions import (
    ConnectTimeoutError,
    FaultIsolationError,
    MaxRetryError,
    NewConnectionError,
    ProtocolError,
    ProxyError,
    ReadTimeoutError,
    SSLError,
)
from urllib3.faultisolation import (
    _FaultIsolationOpen,
    classify_connection_fault,
    create_fault_isolation,
    is_user_cancellation,
)
from urllib3.util.url import parse_url


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def make_policy(
    config: FaultIsolationConfig | None = None,
) -> tuple[FaultIsolationPolicy, FakeClock]:
    clock = FakeClock()
    return (
        FaultIsolationPolicy(config or FaultIsolationConfig(), _time_func=clock),
        clock,
    )


def _policy(pool: HTTPConnectionPool) -> FaultIsolationPolicy:
    assert pool.fault_isolation is not None
    return pool.fault_isolation


def _snapshot(pool: HTTPConnectionPool) -> FaultSnapshot:
    snapshot = pool.fault_isolation_snapshot()
    assert snapshot is not None
    return snapshot


class TestFaultIsolationConfig:
    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            FaultIsolationConfig(failure_threshold=0)
        with pytest.raises(ValueError):
            FaultIsolationConfig(failure_window=0)
        with pytest.raises(ValueError):
            FaultIsolationConfig(recovery_timeout=0)
        with pytest.raises(TypeError):
            FaultIsolationConfig(count_statuses=503)  # type: ignore[arg-type]

    def test_statuses_normalized_to_frozenset(self) -> None:
        config = FaultIsolationConfig(count_statuses=[500, 503])
        assert isinstance(config.count_statuses, frozenset)
        assert config.status_is_counted(503)
        assert not config.status_is_counted(200)
        # Frozen and hashable so it can be part of a PoolKey.
        assert hash(config)

    def test_statuses_callable(self) -> None:
        config = FaultIsolationConfig(count_statuses=lambda status: status >= 500)
        assert config.status_is_counted(503)
        assert not config.status_is_counted(404)

    def test_statuses_disabled_by_default(self) -> None:
        config = FaultIsolationConfig()
        assert not config.status_is_counted(500)


class TestFaultIsolationPolicyStateMachine:
    def test_closed_admits_everybody(self) -> None:
        policy, _ = make_policy(FaultIsolationConfig(failure_threshold=2))
        first = policy.acquire()
        second = policy.acquire()
        assert not first.is_probe
        assert not second.is_probe
        assert policy.state is FaultState.CLOSED

    def test_consecutive_failures_open(self) -> None:
        policy, clock = make_policy(
            FaultIsolationConfig(failure_threshold=3, recovery_timeout=10)
        )
        permits = [policy.acquire() for _ in range(3)]

        permits[0].record_failure(
            FaultReason.CONNECT, NewConnectionError(Mock(), "boom")
        )
        assert policy.state is FaultState.CLOSED
        permits[1].record_failure(FaultReason.CONNECT, ConnectTimeoutError("slow"))
        assert policy.state is FaultState.CLOSED
        permits[2].record_failure(
            FaultReason.PROXY_CONNECT, ProxyError("x", RuntimeError())
        )
        snapshot = policy.snapshot()
        assert snapshot.state is FaultState.OPEN
        assert snapshot.reason == FaultReason.PROXY_CONNECT
        assert snapshot.consecutive_failures == 3
        assert snapshot.total_failures == 3
        assert snapshot.opened_at == clock.value
        assert pytest.approx(snapshot.remaining_recovery_time, abs=0.01) == 10
        assert "ProxyError" in (snapshot.last_error or "")

    def test_open_rejects_with_remaining_time(self) -> None:
        policy, clock = make_policy(
            FaultIsolationConfig(failure_threshold=1, recovery_timeout=5)
        )
        policy.acquire().record_failure(FaultReason.CONNECT, RuntimeError("x"))
        clock.advance(2)
        with pytest.raises(_FaultIsolationOpen) as excinfo:
            policy.acquire()
        assert not excinfo.value.probe_in_flight
        assert pytest.approx(excinfo.value.remaining, abs=0.01) == 3

    def test_success_resets_consecutive_counter(self) -> None:
        policy, _ = make_policy(FaultIsolationConfig(failure_threshold=2))
        p1 = policy.acquire()
        p1.record_failure(FaultReason.CONNECTION_CLOSED, OSError())
        policy.acquire().record_success()
        # Counter restarted: a single failure must not open the state.
        policy.acquire().record_failure(FaultReason.CONNECTION_CLOSED, OSError())
        assert policy.state is FaultState.CLOSED

    def test_half_open_allows_exactly_one_probe(self) -> None:
        policy, clock = make_policy(
            FaultIsolationConfig(failure_threshold=1, recovery_timeout=30)
        )
        policy.acquire().record_failure(FaultReason.TLS_HANDSHAKE, ssl.SSLError())
        clock.advance(30)

        probe = policy.acquire()
        assert probe.is_probe
        assert policy.state is FaultState.HALF_OPEN
        snapshot = policy.snapshot()
        assert snapshot.probe_in_flight
        assert snapshot.state is FaultState.HALF_OPEN

        with pytest.raises(_FaultIsolationOpen) as excinfo:
            policy.acquire()
        assert excinfo.value.probe_in_flight

    def test_probe_success_atomically_clears_state(self) -> None:
        policy, clock = make_policy(
            FaultIsolationConfig(failure_threshold=1, recovery_timeout=100)
        )
        policy.acquire().record_failure(FaultReason.CONNECT, OSError())
        clock.advance(100)
        probe = policy.acquire()
        assert probe.is_probe
        probe.record_success()

        snapshot = policy.snapshot()
        assert snapshot.state is FaultState.CLOSED
        assert not snapshot.probe_in_flight
        assert snapshot.consecutive_failures == 0
        assert snapshot.window_failures == 0
        assert snapshot.recovery_succeeded is True
        assert snapshot.recovered_at == clock.value
        assert snapshot.recovery_error is None

        # Everybody is admitted again.
        assert not policy.acquire().is_probe

    def test_probe_failure_reopens_without_releasing_concurrency(self) -> None:
        policy, clock = make_policy(
            FaultIsolationConfig(failure_threshold=1, recovery_timeout=10)
        )
        policy.acquire().record_failure(FaultReason.CONNECT, OSError("down"))
        clock.advance(10)
        policy.acquire().record_failure(FaultReason.CONNECT, OSError("still down"))

        snapshot = policy.snapshot()
        assert snapshot.state is FaultState.OPEN
        assert snapshot.recovery_succeeded is False
        assert snapshot.recovered_at == clock.value
        assert "still down" in (snapshot.recovery_error or "")
        assert snapshot.remaining_recovery_time > 0
        assert not snapshot.probe_in_flight

        with pytest.raises(_FaultIsolationOpen):
            policy.acquire()

    def test_inflight_success_after_open_clears_state(self) -> None:
        policy, clock = make_policy(
            FaultIsolationConfig(failure_threshold=1, recovery_timeout=10)
        )
        in_flight = policy.acquire()
        policy.acquire().record_failure(FaultReason.CONNECT, OSError())
        assert policy.state is FaultState.OPEN
        clock.advance(2)
        # A request admitted before the state opened completes successfully.
        in_flight.record_success()
        snapshot = policy.snapshot()
        assert snapshot.state is FaultState.CLOSED
        assert snapshot.recovery_succeeded is True

    def test_probe_abandon_reopens_conservatively(self) -> None:
        policy, clock = make_policy(
            FaultIsolationConfig(failure_threshold=1, recovery_timeout=10)
        )
        policy.acquire().record_failure(FaultReason.CONNECT, OSError())
        clock.advance(10)
        opened_at = policy.snapshot().opened_at
        probe = policy.acquire()
        assert probe.is_probe
        probe.abandon()

        snapshot = policy.snapshot()
        assert snapshot.state is FaultState.OPEN
        assert snapshot.opened_at == clock.value
        assert snapshot.opened_at != opened_at
        with pytest.raises(_FaultIsolationOpen):
            policy.acquire()

    def test_late_nonprobe_failure_does_not_renew_cooldown(self) -> None:
        policy, clock = make_policy(
            FaultIsolationConfig(failure_threshold=2, recovery_timeout=10)
        )
        in_flight = policy.acquire()
        other = policy.acquire()
        other.record_failure(FaultReason.CONNECT, OSError())
        policy.acquire().record_failure(FaultReason.CONNECT, OSError())
        assert policy.state is FaultState.OPEN
        opened_at = policy.snapshot().opened_at

        clock.advance(1)
        in_flight.record_failure(FaultReason.CONNECTION_CLOSED, OSError())
        snapshot = policy.snapshot()
        assert snapshot.state is FaultState.OPEN
        assert snapshot.opened_at == opened_at

    def test_window_mode_counts_failures_within_window(self) -> None:
        policy, clock = make_policy(
            FaultIsolationConfig(
                failure_threshold=3, failure_window=10, recovery_timeout=1
            )
        )
        for _ in range(3):
            policy.acquire().record_failure(FaultReason.CONNECT, OSError())
        assert policy.state is FaultState.OPEN
        assert policy.snapshot().window_failures == 3

    def test_window_failures_expire(self) -> None:
        policy, clock = make_policy(
            FaultIsolationConfig(
                failure_threshold=3, failure_window=10, recovery_timeout=1
            )
        )
        policy.acquire().record_failure(FaultReason.CONNECT, OSError())
        clock.advance(6)
        policy.acquire().record_failure(FaultReason.CONNECT, OSError())
        # Ordinary successes do not erase recent window failures...
        policy.acquire().record_success()
        assert policy.snapshot().window_failures == 2
        clock.advance(5)
        # ...but failures outside the window age out.
        assert policy.snapshot().window_failures == 1
        policy.acquire().record_failure(FaultReason.CONNECT, OSError())
        assert policy.snapshot().state is FaultState.CLOSED
        policy.acquire().record_failure(FaultReason.CONNECT, OSError())
        assert policy.snapshot().state is FaultState.OPEN

    def test_recovery_clears_window_history(self) -> None:
        policy, clock = make_policy(
            FaultIsolationConfig(
                failure_threshold=1, failure_window=10, recovery_timeout=1
            )
        )
        policy.acquire().record_failure(FaultReason.CONNECT, OSError())
        clock.advance(1)
        policy.acquire().record_success()
        assert policy.snapshot().window_failures == 0
        assert policy.snapshot().recovery_succeeded is True

    def test_permit_settles_exactly_once(self) -> None:
        policy, _ = make_policy(FaultIsolationConfig(failure_threshold=2))
        permit = policy.acquire()
        permit.record_failure(FaultReason.CONNECT, OSError())
        # A second verdict on the same attempt must be ignored.
        permit.record_success()
        assert policy.snapshot().consecutive_failures == 1
        permit.release()
        assert policy.snapshot().consecutive_failures == 1

    def test_release_without_verdict_abandons_neutrally(self) -> None:
        policy, _ = make_policy(FaultIsolationConfig(failure_threshold=2))
        policy.acquire().release()
        assert policy.snapshot().total_failures == 0

    def test_status_recorded_when_configured(self) -> None:
        policy, _ = make_policy(
            FaultIsolationConfig(failure_threshold=1, count_statuses=frozenset({503}))
        )
        policy.acquire().record_status(503)
        assert policy.state is FaultState.OPEN
        assert policy.snapshot().reason == FaultReason.HTTP_STATUS

    def test_status_treated_as_success_by_default(self) -> None:
        policy, _ = make_policy(FaultIsolationConfig(failure_threshold=1))
        policy.acquire().record_status(500)
        assert policy.state is FaultState.CLOSED

    def test_reset(self) -> None:
        policy, _ = make_policy()
        policy.acquire().record_failure(FaultReason.CONNECT, OSError())
        policy.reset()
        snapshot = policy.snapshot()
        assert snapshot.state is FaultState.CLOSED
        assert snapshot.total_failures == 0
        assert snapshot.reason is None

    def test_thread_safety_many_concurrent_failures(self) -> None:
        policy, _ = make_policy(
            FaultIsolationConfig(failure_threshold=50, recovery_timeout=5)
        )
        recorded = {"count": 0}
        counter_lock = threading.Lock()

        def fail() -> None:
            try:
                permit = policy.acquire()
            except _FaultIsolationOpen:
                return
            permit.record_failure(FaultReason.CONNECT, OSError())
            with counter_lock:
                recorded["count"] += 1

        threads = [threading.Thread(target=fail) for _ in range(100)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        snapshot = policy.snapshot()
        assert recorded["count"] == snapshot.total_failures
        assert recorded["count"] >= 50
        assert snapshot.state is FaultState.OPEN
        assert not snapshot.probe_in_flight
        # Once open, every new attempt is rejected.
        with pytest.raises(_FaultIsolationOpen):
            policy.acquire()


class TestFaultClassification:
    def _config(self, **kwargs: typing.Any) -> FaultIsolationConfig:
        return FaultIsolationConfig(**kwargs)

    def test_proxy_error(self) -> None:
        error = ProxyError("Unable to connect to proxy", OSError())
        assert (
            classify_connection_fault(error, error, self._config())
            == FaultReason.PROXY_CONNECT
        )

    def test_tls_error(self) -> None:
        original = ssl.SSLError("handshake failure")
        wrapped = SSLError(original)
        assert (
            classify_connection_fault(original, wrapped, self._config())
            == FaultReason.TLS_HANDSHAKE
        )

    def test_new_connection_and_timeout(self) -> None:
        config = self._config()
        assert (
            classify_connection_fault(
                NewConnectionError(Mock(), "refused"),
                NewConnectionError(Mock(), "refused"),
                config,
            )
            == FaultReason.CONNECT
        )
        timeout = ConnectTimeoutError("slow")
        assert (
            classify_connection_fault(timeout, timeout, config) == FaultReason.CONNECT
        )

    def test_passive_close(self) -> None:
        original = OSError("Connection reset by peer")
        wrapped = ProtocolError("Connection aborted.", original)
        assert (
            classify_connection_fault(original, wrapped, self._config())
            == FaultReason.CONNECTION_CLOSED
        )
        remote = RemoteDisconnected("Remote end closed connection")
        assert isinstance(remote, OSError)
        assert (
            classify_connection_fault(
                remote, ProtocolError("Connection aborted.", remote), self._config()
            )
            == FaultReason.CONNECTION_CLOSED
        )

    def test_read_timeout_opt_in(self) -> None:
        error = ReadTimeoutError(Mock(), "/", "read timed out")
        assert classify_connection_fault(error, error, self._config()) is None
        enabled = self._config(count_read_errors=True)
        assert (
            classify_connection_fault(error, error, enabled) == FaultReason.READ_ERROR
        )

    def test_incomplete_read_opt_in(self) -> None:
        error = IncompleteRead(b'', 1)
        assert (
            classify_connection_fault(error, ProtocolError("x", error), self._config())
            is None
        )
        enabled = self._config(count_read_errors=True)
        assert (
            classify_connection_fault(error, ProtocolError("x", error), enabled)
            == FaultReason.READ_ERROR
        )

    def test_unknown_error_not_counted(self) -> None:
        assert (
            classify_connection_fault(
                RuntimeError("?"), RuntimeError("?"), self._config()
            )
            is None
        )

    def test_user_cancellation_predicate(self) -> None:
        assert is_user_cancellation(KeyboardInterrupt())
        assert is_user_cancellation(asyncio.CancelledError())
        assert not is_user_cancellation(RuntimeError())
        assert not is_user_cancellation(SystemExit())


class TestFaultIsolationFactory:
    def test_disabled(self) -> None:
        assert create_fault_isolation(None) is None
        assert create_fault_isolation(False) is None

    def test_enabled_defaults(self) -> None:
        policy = create_fault_isolation(True)
        assert isinstance(policy, FaultIsolationPolicy)
        assert policy.config == FaultIsolationConfig()

    def test_config_creates_independent_policy(self) -> None:
        config = FaultIsolationConfig(failure_threshold=3)
        first = create_fault_isolation(config)
        second = create_fault_isolation(config)
        assert first is not second
        assert first is not None and first.config is config

    def test_policy_passthrough(self) -> None:
        policy = FaultIsolationPolicy()
        assert create_fault_isolation(policy) is policy

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError):
            create_fault_isolation("yes")  # type: ignore[arg-type]


@pytest.fixture
def mock_connection() -> Mock:
    conn = Mock()
    conn.proxy = None
    conn.is_closed = True
    return conn


class TestPoolFaultIntegration:
    def test_disabled_by_default(self) -> None:
        pool = HTTPConnectionPool("localhost", retries=False)
        assert pool.fault_isolation is None
        assert pool.fault_isolation_snapshot() is None
        pool.close()

    def test_connect_failures_open_and_then_fail_fast(
        self, mock_connection: Mock
    ) -> None:
        pool = HTTPConnectionPool(
            "localhost",
            retries=False,
            fault_isolation=FaultIsolationConfig(
                failure_threshold=2, recovery_timeout=30
            ),
        )
        with (
            patch.object(pool, "_get_conn", return_value=mock_connection),
            patch.object(pool, "_put_conn"),
            patch.object(
                pool, "_make_request", side_effect=OSError("connection reset")
            ),
        ):
            for _ in range(2):
                with pytest.raises(ProtocolError):
                    pool.request("GET", "/")

        snapshot = _snapshot(pool)
        assert snapshot is not None
        assert snapshot.state is FaultState.OPEN
        assert snapshot.reason == FaultReason.CONNECTION_CLOSED
        assert snapshot.remaining_recovery_time > 0

        # While open, no connection may be touched.
        with patch.object(
            pool, "_get_conn", side_effect=AssertionError("queue touched")
        ):
            with patch.object(
                pool, "_make_request", side_effect=AssertionError("connection touched")
            ):
                with pytest.raises(FaultIsolationError) as excinfo:
                    pool.request("GET", "/")
        assert not excinfo.value.probe_in_flight
        assert excinfo.value.remaining_recovery_time is not None
        pool.close()

    def test_retry_wraps_rejection_in_max_retry_error(
        self, mock_connection: Mock
    ) -> None:
        pool = HTTPConnectionPool(
            "localhost",
            retries=3,
            fault_isolation=FaultIsolationConfig(
                failure_threshold=5, recovery_timeout=30
            ),
        )
        with (
            patch.object(pool, "_get_conn", return_value=mock_connection),
            patch.object(pool, "_put_conn"),
            patch.object(pool, "_make_request", side_effect=OSError("reset")),
        ):
            # 4 failed attempts: stays closed (threshold 5), ordinary MaxRetryError.
            with pytest.raises(MaxRetryError) as first:
                pool.request("GET", "/")
            assert not isinstance(first.value.reason, FaultIsolationError)
            assert _snapshot(pool).total_failures == 4

            # Attempt 5 opens the state, attempts 6+ are rejected without
            # touching a connection and exhaust retries immediately.
            with pytest.raises(MaxRetryError) as second:
                pool.request("GET", "/")
        assert isinstance(second.value.reason, FaultIsolationError)
        assert _snapshot(pool).total_failures == 5
        pool.close()

    def test_successful_probe_recovers(self, mock_connection: Mock) -> None:
        pool = HTTPConnectionPool(
            "localhost",
            retries=False,
            fault_isolation=FaultIsolationConfig(
                failure_threshold=1, recovery_timeout=0.05
            ),
        )
        with (
            patch.object(pool, "_get_conn", return_value=mock_connection),
            patch.object(pool, "_put_conn"),
            patch.object(pool, "_make_request", side_effect=OSError("reset")),
        ):
            with pytest.raises(ProtocolError):
                pool.request("GET", "/")
        assert _snapshot(pool).state is FaultState.OPEN

        time.sleep(0.06)
        response = Mock(status=200)
        with (
            patch.object(pool, "_get_conn", return_value=mock_connection),
            patch.object(pool, "_put_conn"),
            patch.object(pool, "_make_request", return_value=response),
        ):
            recovered = pool.urlopen("GET", "/", retries=False, redirect=False)
        assert recovered.status == 200
        snapshot = _snapshot(pool)
        assert snapshot.state is FaultState.CLOSED
        assert snapshot.recovery_succeeded is True
        assert snapshot.consecutive_failures == 0
        pool.close()

    def test_configured_status_code_opens_state(self, mock_connection: Mock) -> None:
        pool = HTTPConnectionPool(
            "localhost",
            retries=False,
            fault_isolation=FaultIsolationConfig(
                failure_threshold=1, count_statuses=frozenset({503})
            ),
        )
        with (
            patch.object(pool, "_get_conn", return_value=mock_connection),
            patch.object(pool, "_put_conn"),
            patch.object(pool, "_make_request", return_value=Mock(status=503)),
        ):
            response = pool.urlopen("GET", "/", retries=False, redirect=False)
        assert response.status == 503
        snapshot = _snapshot(pool)
        assert snapshot.state is FaultState.OPEN
        assert snapshot.reason == FaultReason.HTTP_STATUS

        # A non-configured status code is a healthy connection and clears it.
        pool2 = HTTPConnectionPool(
            "localhost",
            retries=False,
            fault_isolation=FaultIsolationConfig(
                failure_threshold=1, count_statuses=frozenset({503})
            ),
        )
        with (
            patch.object(pool2, "_get_conn", return_value=mock_connection),
            patch.object(pool2, "_put_conn"),
            patch.object(pool2, "_make_request", return_value=Mock(status=500)),
        ):
            pool2.urlopen("GET", "/", retries=False, redirect=False)
        assert _snapshot(pool2).state is FaultState.CLOSED
        pool.close()
        pool2.close()

    def test_read_errors_are_not_counted_by_default(
        self, mock_connection: Mock
    ) -> None:
        pool = HTTPConnectionPool(
            "localhost",
            retries=False,
            fault_isolation=FaultIsolationConfig(failure_threshold=1),
        )
        with (
            patch.object(pool, "_get_conn", return_value=mock_connection),
            patch.object(pool, "_put_conn"),
            patch.object(
                pool,
                "_make_request",
                side_effect=ReadTimeoutError(pool, "/", "read timed out"),
            ),
        ):
            with pytest.raises(ReadTimeoutError):
                pool.request("GET", "/")
        assert _snapshot(pool).state is FaultState.CLOSED
        pool.close()

    def test_read_errors_count_when_enabled(self, mock_connection: Mock) -> None:
        pool = HTTPConnectionPool(
            "localhost",
            retries=False,
            fault_isolation=FaultIsolationConfig(
                failure_threshold=1, count_read_errors=True
            ),
        )
        with (
            patch.object(pool, "_get_conn", return_value=mock_connection),
            patch.object(pool, "_put_conn"),
            patch.object(
                pool,
                "_make_request",
                side_effect=ReadTimeoutError(pool, "/", "read timed out"),
            ),
        ):
            with pytest.raises(ReadTimeoutError):
                pool.request("GET", "/")
        snapshot = _snapshot(pool)
        assert snapshot.state is FaultState.OPEN
        assert snapshot.reason == FaultReason.READ_ERROR
        pool.close()

    def test_tls_handshake_failure_classified(self, mock_connection: Mock) -> None:
        pool = HTTPSConnectionPool(
            "localhost",
            retries=False,
            fault_isolation=FaultIsolationConfig(failure_threshold=1),
        )
        with (
            patch.object(pool, "_get_conn", return_value=mock_connection),
            patch.object(pool, "_put_conn"),
            patch.object(pool, "_make_request", side_effect=ssl.SSLError("bad cert")),
        ):
            with pytest.raises(SSLError):
                pool.request("GET", "/")
        snapshot = _snapshot(pool)
        assert snapshot.state is FaultState.OPEN
        assert snapshot.reason == FaultReason.TLS_HANDSHAKE
        pool.close()

    def test_proxy_failure_classified(self) -> None:
        mock_conn = Mock()
        mock_conn.proxy = Mock(scheme="http")
        mock_conn.has_connected_to_proxy = False
        pool = HTTPConnectionPool(
            "localhost",
            retries=False,
            _proxy=parse_url("http://proxy:8080"),
            fault_isolation=FaultIsolationConfig(failure_threshold=1),
        )
        with (
            patch.object(pool, "_get_conn", return_value=mock_conn),
            patch.object(pool, "_put_conn"),
            patch.object(pool, "_make_request", side_effect=OSError("proxy refused")),
        ):
            with pytest.raises(ProxyError):
                pool.urlopen(
                    "GET",
                    "http://example.com/",
                    retries=False,
                    redirect=False,
                    assert_same_host=False,
                )
        snapshot = _snapshot(pool)
        assert snapshot.state is FaultState.OPEN
        assert snapshot.reason == FaultReason.PROXY_CONNECT
        pool.close()

    def test_cancellation_counts_only_when_enabled(self, mock_connection: Mock) -> None:
        enabled = HTTPConnectionPool(
            "localhost",
            retries=False,
            fault_isolation=FaultIsolationConfig(
                failure_threshold=1, count_cancelled=True
            ),
        )
        with (
            patch.object(enabled, "_get_conn", return_value=mock_connection),
            patch.object(enabled, "_put_conn"),
            patch.object(enabled, "_make_request", side_effect=KeyboardInterrupt()),
        ):
            with pytest.raises(KeyboardInterrupt):
                enabled.request("GET", "/")
        assert _snapshot(enabled).reason == FaultReason.CANCELLED
        enabled.close()

        disabled = HTTPConnectionPool(
            "localhost",
            retries=False,
            fault_isolation=FaultIsolationConfig(failure_threshold=1),
        )
        with (
            patch.object(disabled, "_get_conn", return_value=mock_connection),
            patch.object(disabled, "_put_conn"),
            patch.object(disabled, "_make_request", side_effect=KeyboardInterrupt()),
        ):
            with pytest.raises(KeyboardInterrupt):
                disabled.request("GET", "/")
        assert _snapshot(disabled).total_failures == 0
        disabled.close()

    def test_open_state_keeps_pooled_connections_untouched(self) -> None:
        pool = HTTPConnectionPool(
            "localhost",
            maxsize=2,
            block=True,
            retries=False,
            fault_isolation=FaultIsolationConfig(
                failure_threshold=1, recovery_timeout=30
            ),
        )
        # Check out both tokens (each None token is replaced by a fresh
        # connection object), then return both connections to the idle queue.
        first = pool._get_conn()
        second = pool._get_conn()
        with patch.object(second, "close") as second_close:
            pool._put_conn(first)
            pool._put_conn(second)
            assert pool.pool is not None
            assert pool.pool.qsize() == 2

            assert pool.fault_isolation is not None
            _policy(pool).acquire().record_failure(
                FaultReason.CONNECT, OSError()
            )
            assert _snapshot(pool).state is FaultState.OPEN
            assert pool.pool.qsize() == 2

            # Fail-fast must not block on pool_timeout or consume a token.
            started = time.monotonic()
            for _ in range(5):
                with pytest.raises(FaultIsolationError):
                    pool.request("GET", "/", pool_timeout=30)
            assert time.monotonic() - started < 1
            assert pool.pool.qsize() == 2
            second_close.assert_not_called()
        pool.close()

    def test_waiting_request_admitted_before_open_is_not_rejected(self) -> None:
        pool = HTTPConnectionPool(
            "localhost",
            maxsize=1,
            block=True,
            retries=False,
            fault_isolation=FaultIsolationConfig(
                failure_threshold=1, recovery_timeout=30
            ),
        )
        # Occupy the single queue token outside the breaker gate.
        pool._get_conn()

        result: dict[str, typing.Any] = {}

        def waiter() -> None:
            try:
                response = pool.urlopen(
                    "GET", "/", retries=False, redirect=False, pool_timeout=5
                )
                result["response"] = response.status
            except BaseException as e:  # noqa: BLE001
                result["error"] = e

        mock_conn = Mock()
        mock_conn.proxy = None
        mock_conn.is_closed = True
        thread = threading.Thread(target=waiter)
        with patch.object(pool, "_make_request", return_value=Mock(status=200)):
            thread.start()
            time.sleep(0.1)
            # Another caller opens the state while the waiter is queued.
            _policy(pool).acquire().record_failure(
                FaultReason.CONNECT, OSError()
            )
            time.sleep(0.05)
            # Release the token; the queued waiter proceeds despite OPEN state.
            pool._put_conn(None)
            thread.join(2)

        assert not thread.is_alive()
        assert result == {"response": 200}
        assert _snapshot(pool).state is FaultState.CLOSED
        pool.close()

    def test_close_resets_fault_state(self) -> None:
        pool = HTTPConnectionPool(
            "localhost",
            fault_isolation=FaultIsolationConfig(failure_threshold=1),
        )
        _policy(pool).acquire().record_failure(FaultReason.CONNECT, OSError())
        assert _snapshot(pool).state is FaultState.OPEN
        pool.close()
        assert _snapshot(pool).state is FaultState.CLOSED

    def test_single_probe_under_concurrency(self, mock_connection: Mock) -> None:
        pool = HTTPConnectionPool(
            "localhost",
            maxsize=8,
            retries=False,
            fault_isolation=FaultIsolationConfig(
                failure_threshold=1, recovery_timeout=0.02
            ),
        )
        lock = threading.Lock()
        inflight = {"current": 0, "peak": 0}
        attempts = {"count": 0}

        def always_fails(*args: typing.Any, **kwargs: typing.Any) -> None:
            with lock:
                inflight["current"] += 1
                inflight["peak"] = max(inflight["peak"], inflight["current"])
                attempts["count"] += 1
            time.sleep(0.005)
            with lock:
                inflight["current"] -= 1
            raise OSError("still down")

        errors: list[BaseException] = []

        def worker() -> None:
            deadline = time.monotonic() + 0.6
            while time.monotonic() < deadline:
                try:
                    pool.urlopen("GET", "/", retries=False, redirect=False)
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        with (
            patch.object(pool, "_get_conn", return_value=mock_connection),
            patch.object(pool, "_put_conn"),
            patch.object(pool, "_make_request", side_effect=always_fails),
        ):
            # Prime the open state before letting workers hammer the pool.
            with pytest.raises(ProtocolError):
                pool.urlopen("GET", "/", retries=False, redirect=False)
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert attempts["count"] >= 2
        assert inflight["peak"] == 1
        assert errors
        # Rejected attempts fail fast; the occasional admitted probe dies as a
        # connection protocol error, never concurrently with another probe.
        assert all(isinstance(e, (FaultIsolationError, ProtocolError)) for e in errors)
        assert _snapshot(pool).state is FaultState.OPEN
        pool.close()


class TestPoolManagerFaultIntegration:
    def test_each_subpool_keeps_independent_state(self) -> None:
        config = FaultIsolationConfig(failure_threshold=1)
        manager = PoolManager(fault_isolation=config, retries=False)
        pool_a = manager.connection_from_url("http://a.example/")
        pool_b = manager.connection_from_url("http://b.example/")
        assert pool_a.fault_isolation is not None
        assert pool_b.fault_isolation is not None
        assert pool_a.fault_isolation is not pool_b.fault_isolation
        assert _policy(pool_a).config == _policy(pool_b).config

        mock_conn = Mock(proxy=None, is_closed=True)
        with (
            patch.object(pool_a, "_get_conn", return_value=mock_conn),
            patch.object(pool_a, "_put_conn"),
            patch.object(pool_a, "_make_request", side_effect=OSError("a down")),
        ):
            with pytest.raises(ProtocolError):
                manager.request("GET", "http://a.example/", retries=False)

        snapshots = manager.fault_isolation_snapshots()
        assert snapshots["http://a.example:80"].state is FaultState.OPEN
        assert snapshots["http://b.example:80"].state is FaultState.CLOSED

        # The healthy target keeps serving.
        response = Mock(status=200)
        response.get_redirect_location.return_value = None
        with (
            patch.object(pool_b, "_get_conn", return_value=mock_conn),
            patch.object(pool_b, "_put_conn"),
            patch.object(pool_b, "_make_request", return_value=response),
        ):
            result = manager.request("GET", "http://b.example/", retries=False)
        assert result.status == 200
        manager.clear()

    def test_policy_instance_is_propagated_as_config(self) -> None:
        shared_policy = FaultIsolationPolicy(FaultIsolationConfig(failure_threshold=2))
        manager = PoolManager(fault_isolation=shared_policy)
        pool = manager.connection_from_url("http://a.example/")
        assert pool.fault_isolation is not shared_policy
        assert _policy(pool).config == shared_policy.config
        manager.clear()

    def test_dynamic_subpools_share_key_for_same_config(self) -> None:
        config = FaultIsolationConfig(failure_threshold=2)
        manager = PoolManager(fault_isolation=config)
        first = manager.connection_from_url("http://a.example/")
        second = manager.connection_from_url(
            "http://a.example/", pool_kwargs={"fault_isolation": config}
        )
        assert first is second

        other_config = FaultIsolationConfig(failure_threshold=3)
        third = manager.connection_from_url(
            "http://a.example/", pool_kwargs={"fault_isolation": other_config}
        )
        assert third is not first
        manager.clear()

    def test_clear_discards_fault_state(self) -> None:
        manager = PoolManager(
            fault_isolation=FaultIsolationConfig(failure_threshold=1), retries=False
        )
        pool = manager.connection_from_url("http://a.example/")
        _policy(pool).acquire().record_failure(FaultReason.CONNECT, OSError())
        assert _snapshot(pool).state is FaultState.OPEN
        manager.clear()
        assert manager.fault_isolation_snapshots() == {}

        fresh = manager.connection_from_url("http://a.example/")
        assert fresh is not pool
        assert _snapshot(fresh).state is FaultState.CLOSED
        manager.clear()

    def test_evicted_pool_disappears_from_snapshots(self) -> None:
        manager = PoolManager(
            num_pools=1, fault_isolation=FaultIsolationConfig(failure_threshold=1)
        )
        pool_a = manager.connection_from_url("http://a.example/")
        _policy(pool_a).acquire().record_failure(FaultReason.CONNECT, OSError())
        pool_b = manager.connection_from_url("http://b.example/")
        _policy(pool_b).acquire().record_failure(FaultReason.CONNECT, OSError())

        snapshots = manager.fault_isolation_snapshots()
        assert "http://b.example:80" in snapshots
        assert "http://a.example:80" not in snapshots
        manager.clear()

    def test_proxy_manager_propagates_policy(self) -> None:
        manager = ProxyManager(
            "http://localhost:3128",
            fault_isolation=FaultIsolationConfig(failure_threshold=2),
        )
        http_pool = manager.connection_from_url("http://a.example/")
        https_pool = manager.connection_from_url("https://b.example/")
        assert http_pool.fault_isolation is not None
        assert https_pool.fault_isolation is not None
        assert http_pool.fault_isolation is not https_pool.fault_isolation
        assert http_pool.proxy == parse_url("http://localhost:3128")
        manager.clear()

    def test_disabled_manager_has_no_snapshots(self) -> None:
        manager = PoolManager()
        manager.connection_from_url("http://a.example/")
        assert manager.fault_isolation_snapshots() == {}
        manager.clear()

    def test_retry_object_unchanged_in_connection_pool_kwargs(self) -> None:
        manager = PoolManager(fault_isolation=True, retries=Retry(5))
        pool = manager.connection_from_url("http://a.example/")
        assert isinstance(pool.retries, Retry)
        assert pool.retries.total == 5
        manager.clear()
